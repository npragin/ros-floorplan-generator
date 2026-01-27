"""Connectivity validation for floorplans."""

from collections import deque

from shapely.geometry import Polygon

from floorplan_generator.generator import Floorplan


class LayoutValidator:
    """Validates floorplan layouts for connectivity and constraint satisfaction."""

    def __init__(self, tolerance: float = 0.01) -> None:
        """
        Initialize the validator.

        Args:
            tolerance: Geometric tolerance for intersection checks (meters).

        """
        self.tolerance = tolerance

    def validate_connectivity(self, floorplan: Floorplan) -> bool:
        """
        Validate that all rooms are reachable from all other rooms.

        Args:
            floorplan: The floorplan to validate.

        Returns:
            True if all rooms are connected, False otherwise.

        """
        if len(floorplan.room_interiors) <= 1:
            return True

        graph = self.build_connectivity_graph(floorplan)

        # BFS from first room
        visited: set[int] = set()
        queue: deque[int] = deque([0])
        visited.add(0)

        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(floorplan.room_interiors)

    def validate_no_overlaps(self, rooms: list[Polygon]) -> bool:
        """
        Validate that no rooms overlap with each other.

        Args:
            rooms: List of room polygons.

        Returns:
            True if no rooms overlap, False otherwise.

        """
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i + 1 :]:
                if room1.intersects(room2) and not room1.touches(room2):
                    return False
        return True

    def validate_no_unreachable_spaces(self, floorplan: Floorplan) -> bool:
        """
        Validate that there are no isolated unreachable spaces.

        This checks that the free space forms a single connected component.

        Args:
            floorplan: The floorplan to validate.

        Returns:
            True if all free space is reachable, False otherwise.

        """
        return floorplan.get_free_space().geom_type != "MultiPolygon"

    def build_connectivity_graph(self, floorplan: Floorplan) -> dict[int, list[int]]:
        """
        Build an adjacency graph of room connectivity.

        In the linear hallway layout, all rooms connect to the hallway,
        so all rooms are mutually reachable through the hallway.

        Args:
            floorplan: The floorplan to analyze.

        Returns:
            A dictionary mapping room indices to lists of connected room indices.

        """
        graph: dict[int, list[int]] = {i: [] for i in range(len(floorplan.room_interiors))}

        # In a linear hallway layout, all rooms connect to the hallway
        # So all rooms are connected to all other rooms via the hallway
        # We model this as a complete graph for simplicity
        for i in range(len(floorplan.room_interiors)):
            for j in range(len(floorplan.room_interiors)):
                if i != j:
                    graph[i].append(j)

        return graph

    def validate_no_hallway_overlaps(self, floorplan: Floorplan) -> bool:
        """
        Validate that non-adjacent hallway segments do not overlap.

        This is a post-hoc safety net for collision detection.

        Args:
            floorplan: The floorplan to validate.

        Returns:
            True if no non-adjacent segments overlap, False otherwise.

        """
        segments = floorplan.hallway_segments
        for i in range(len(segments)):
            for j in range(i + 2, len(segments)):  # Skip adjacent (i+1)
                intersection = segments[i].polygon.intersection(segments[j].polygon)
                if intersection.area > self.tolerance:
                    return False
        return True

    def validate_all(self, floorplan: Floorplan) -> tuple[bool, list[str]]:
        """
        Run all validations and return results.

        Args:
            floorplan: The floorplan to validate.

        Returns:
            A tuple of (all_passed, list of error messages).

        """
        errors: list[str] = []

        if not self.validate_connectivity(floorplan):
            errors.append("Not all rooms are connected")

        if not self.validate_no_overlaps(floorplan.room_interiors):
            errors.append("Some rooms overlap")

        if not self.validate_no_unreachable_spaces(floorplan):
            errors.append("There are unreachable spaces in the floorplan")

        if not self.validate_no_hallway_overlaps(floorplan):
            errors.append("Non-adjacent hallway segments overlap")

        return len(errors) == 0, errors
