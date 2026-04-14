"""Robot spawn position generation via greedy circle packing."""

from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def generate_spawn_positions(
    free_space: Polygon | MultiPolygon,
    num_robots: int,
    robot_radius: float,
    min_clearance: float,
    resolution: float = 0.05,
) -> list[tuple[float, float]]:
    """
    Generate spawn positions for robots using greedy circle packing.

    Erodes the free space by (robot_radius + min_clearance) to get the valid
    region for robot centers, then greedily places robots as close together
    as possible starting from the centroid.

    Args:
        free_space: The navigable space polygon.
        num_robots: Number of robots to place.
        robot_radius: Radius of each robot in meters.
        min_clearance: Minimum clearance from walls and between robot edges.
        resolution: Grid resolution for candidate points in meters.
        debug_path: If provided, saves a debug image showing the valid
            spawn region and placed robot positions.

    Returns:
        List of (x, y) positions in world coordinates.

    Raises:
        ValueError: If not enough valid positions can be found.

    """
    # Erode free space so robot edges are at least min_clearance from walls
    erosion = robot_radius + min_clearance
    valid_region = free_space.buffer(-erosion)

    if valid_region.is_empty:
        raise ValueError(
            f"No valid spawn region after eroding free space by {erosion}m "
            f"(robot_radius={robot_radius} + min_clearance={min_clearance})"
        )

    # Minimum center-to-center distance between robots
    min_center_dist = min_clearance + 2 * robot_radius

    # Build grid of candidate points within the valid region
    minx, miny, maxx, maxy = valid_region.bounds
    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])

    # Filter to points inside the valid region
    from shapely.vectorized import contains

    mask = contains(valid_region, candidates[:, 0], candidates[:, 1])
    candidates = candidates[mask]

    if len(candidates) == 0:
        raise ValueError("No candidate points found within the valid spawn region")

    import random

    # Pick a random starting point from the valid candidates
    start = candidates[random.randrange(len(candidates))]

    # Greedy placement
    positions: list[tuple[float, float]] = [(float(start[0]), float(start[1]))]
    placed = np.array([start])

    for _ in range(1, num_robots):
        # Compute distance from each candidate to all placed robots
        dists_to_placed = np.linalg.norm(candidates[:, np.newaxis, :] - placed[np.newaxis, :, :], axis=2)
        # Minimum distance to any placed robot for each candidate
        min_dists = dists_to_placed.min(axis=1)

        # Filter candidates that satisfy spacing constraint
        valid_mask = min_dists >= min_center_dist
        valid_candidates = candidates[valid_mask]

        if len(valid_candidates) == 0:
            raise ValueError(
                f"Could only place {len(positions)}/{num_robots} robots. "
                f"Not enough space with robot_radius={robot_radius} and "
                f"min_clearance={min_clearance}"
            )

        # Find closest valid point to cluster centroid
        cluster_centroid = placed.mean(axis=0)
        dists_to_centroid = np.linalg.norm(valid_candidates - cluster_centroid, axis=1)
        best_idx = np.argmin(dists_to_centroid)

        new_pos = valid_candidates[best_idx]
        positions.append((float(new_pos[0]), float(new_pos[1])))
        placed = np.vstack([placed, new_pos])

    return positions


def generate_extra_point_positions(
    free_space: Polygon | MultiPolygon,
    num_extra_points: int,
    extra_point_radius: float,
    min_clearance: float,
    existing_positions: list[tuple[float, float]],
    robot_radius: float,
    resolution: float = 0.05,
) -> list[tuple[float, float]]:
    """
    Generate extra point positions randomly distributed across free space.

    Places points one at a time at random valid locations, maintaining
    clearance from walls, robots, and other extra points.

    Args:
        free_space: The navigable space polygon.
        num_extra_points: Number of extra points to place.
        extra_point_radius: Radius of each extra point in meters.
        min_clearance: Minimum clearance from walls and between extra point edges.
        existing_positions: List of (x, y) robot positions to avoid.
        robot_radius: Radius of existing robots (used for clearance calculation).
        resolution: Grid resolution for candidate points in meters.

    Returns:
        List of (x, y) positions in world coordinates.

    Raises:
        ValueError: If not enough valid positions can be found.

    """
    import random as rng

    # Erode free space so extra point edges are at least min_clearance from walls
    erosion = extra_point_radius + min_clearance
    valid_region = free_space.buffer(-erosion)

    if valid_region.is_empty:
        raise ValueError(
            f"No valid region after eroding free space by {erosion}m "
            f"(extra_point_radius={extra_point_radius} + min_clearance={min_clearance})"
        )

    # Minimum center-to-center distance between extra points
    min_center_dist = min_clearance + 2 * extra_point_radius

    # Minimum center-to-center distance from robots
    min_robot_dist = min_clearance + robot_radius + extra_point_radius

    # Build grid of candidate points within the valid region
    minx, miny, maxx, maxy = valid_region.bounds
    xs = np.arange(minx, maxx + resolution, resolution)
    ys = np.arange(miny, maxy + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])

    # Filter to points inside the valid region
    from shapely.vectorized import contains

    mask = contains(valid_region, candidates[:, 0], candidates[:, 1])
    candidates = candidates[mask]

    if len(candidates) == 0:
        raise ValueError("No candidate points found within the valid extra point region")

    # Filter candidates to maintain clearance from existing robot positions
    if existing_positions:
        existing = np.array(existing_positions)
        dists_to_existing = np.linalg.norm(candidates[:, np.newaxis, :] - existing[np.newaxis, :, :], axis=2)
        min_dists_to_existing = dists_to_existing.min(axis=1)
        candidates = candidates[min_dists_to_existing >= min_robot_dist]

    if len(candidates) == 0:
        raise ValueError("No candidate points found after filtering for robot clearance")

    # Random placement
    positions: list[tuple[float, float]] = []
    placed = np.empty((0, 2))

    for i in range(num_extra_points):
        if i > 0:
            # Filter candidates that satisfy spacing from already-placed extra points
            dists_to_placed = np.linalg.norm(candidates[:, np.newaxis, :] - placed[np.newaxis, :, :], axis=2)
            min_dists = dists_to_placed.min(axis=1)
            valid_mask = min_dists >= min_center_dist
            candidates = candidates[valid_mask]

        if len(candidates) == 0:
            raise ValueError(
                f"Could only place {len(positions)}/{num_extra_points} extra points. "
                f"Not enough space with extra_point_radius={extra_point_radius} and "
                f"min_clearance={min_clearance}"
            )

        idx = rng.randrange(len(candidates))
        new_pos = candidates[idx]
        positions.append((float(new_pos[0]), float(new_pos[1])))
        placed = np.vstack([placed, new_pos]) if len(placed) > 0 else np.array([new_pos])

    return positions


def transform_to_map_center(
    positions: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """
    Transform positions so the map center is at the origin.

    Args:
        positions: List of (x, y) positions in world coordinates.
        bounds: Map bounds as (min_x, min_y, max_x, max_y).

    Returns:
        List of (x, y) positions centered at map origin.

    """
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    return [(x - center_x, y - center_y) for x, y in positions]


def write_spawn_yaml(
    positions: list[tuple[float, float]] | None,
    map_width: float,
    map_height: float,
    output_path: Path,
    extra_points: list[tuple[float, float]] | None = None,
) -> None:
    """
    Write spawn positions and map dimensions to a YAML file.

    Args:
        positions: List of (x, y) robot positions, or None if no robots.
        map_width: Width of the generated map in meters.
        map_height: Height of the generated map in meters.
        output_path: Path to the output YAML file.
        extra_points: Optional list of (x, y) extra point positions.

    """
    import yaml

    data: dict = {
        "map": {
            "width": round(map_width, 4),
            "height": round(map_height, 4),
        },
    }

    if positions is not None:
        data["robots"] = [{"robot_id": i, "x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(positions)]

    if extra_points is not None:
        data["extra_points"] = [
            {"point_id": i, "x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(extra_points)
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
