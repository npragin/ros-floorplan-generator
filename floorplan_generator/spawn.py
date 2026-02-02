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

    # Find starting point: centroid of valid region, or nearest valid point
    centroid = valid_region.centroid
    if valid_region.contains(centroid):
        start = np.array([centroid.x, centroid.y])
    else:
        # Find nearest candidate to centroid
        dists = np.linalg.norm(candidates - np.array([centroid.x, centroid.y]), axis=1)
        start = candidates[np.argmin(dists)]

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
    positions: list[tuple[float, float]],
    map_width: float,
    map_height: float,
    output_path: Path,
) -> None:
    """
    Write spawn positions and map dimensions to a YAML file.

    Args:
        positions: List of (x, y) positions.
        map_width: Width of the generated map in meters.
        map_height: Height of the generated map in meters.
        output_path: Path to the output YAML file.

    """
    import yaml

    data = {
        "map": {
            "width": round(map_width, 4),
            "height": round(map_height, 4),
        },
        "robots": [{"robot_id": i, "x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(positions)],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
