"""Geometric helper functions using Shapely."""

from dataclasses import dataclass
from typing import Literal

from shapely.geometry import Polygon, box

Direction = Literal["east", "west", "north", "south"]


@dataclass
class HallwaySegment:
    """
    Represents a single segment of a hallway.

    Attributes:
        start: Starting point (x, y) of the segment centerline.
        end: Ending point (x, y) of the segment centerline.
        direction: Direction the hallway runs ("east", "west", "north", "south").
        polygon: The Shapely Polygon representing the segment interior.

    """

    start: tuple[float, float]
    end: tuple[float, float]
    direction: Direction
    polygon: Polygon
    is_open_space: bool = False

    @property
    def length(self) -> float:
        """Get the length of this segment."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx**2 + dy**2) ** 0.5

    @property
    def is_horizontal(self) -> bool:
        """Check if this segment runs horizontally (east-west)."""
        return self.direction in ("east", "west")

    @property
    def is_vertical(self) -> bool:
        """Check if this segment runs vertically (north-south)."""
        return self.direction in ("north", "south")


def create_rectangle(x: float, y: float, width: float, height: float) -> Polygon:
    """
    Create a rectangle polygon.

    Args:
        x: X-coordinate of the bottom-left corner.
        y: Y-coordinate of the bottom-left corner.
        width: Width of the rectangle (along x-axis).
        height: Height of the rectangle (along y-axis).

    Returns:
        A Shapely Polygon representing the rectangle.

    """
    return box(x, y, x + width, y + height)


def create_room(center_x: float, center_y: float, side_length: float):
    """
    Create a square room centered at the given coordinates.

    Args:
        center_x: X-coordinate of the room center.
        center_y: Y-coordinate of the room center.
        side_length: Length of each side of the square room.

    Returns:
        A Shapely Polygon representing the room.

    """
    half_side = side_length / 2
    return box(
        center_x - half_side,
        center_y - half_side,
        center_x + half_side,
        center_y + half_side,
    )


def create_hallway(start_x: float, start_y: float, length: float, width: float) -> Polygon:
    """
    Create a horizontal hallway polygon.

    Args:
        start_x: X-coordinate of the left edge of the hallway.
        start_y: Y-coordinate of the center of the hallway.
        length: Length of the hallway (along x-axis).
        width: Width of the hallway (along y-axis).

    Returns:
        A Shapely Polygon representing the hallway.

    """
    half_width = width / 2
    return box(start_x, start_y - half_width, start_x + length, start_y + half_width)


def create_door_opening(
    x_center: float,
    y_center: float,
    door_width: float,
    wall_thickness: float,
    is_horizontal: bool = True,
) -> Polygon:
    """
    Create a door opening as a polygon (for subtracting from walls).

    Args:
        x_center: X-coordinate of the door center.
        y_center: Y-coordinate of the door center.
        door_width: Width of the door opening.
        wall_thickness: Thickness of the wall (determines door depth).
        is_horizontal: If True, door is horizontal (passage along y-axis).

    Returns:
        A Shapely Polygon representing the door opening.

    """
    if is_horizontal:
        # Door spans horizontally, passage is vertical
        return box(
            x_center - door_width / 2,
            y_center - wall_thickness / 2,
            x_center + door_width / 2,
            y_center + wall_thickness / 2,
        )
    else:
        # Door spans vertically, passage is horizontal
        return box(
            x_center - wall_thickness / 2,
            y_center - door_width / 2,
            x_center + wall_thickness / 2,
            y_center + door_width / 2,
        )


def get_wall_center_on_side(
    room: Polygon,
    side: str,
) -> tuple[float, float]:
    """
    Get the center point of a specific side of a room.

    Args:
        room: The room polygon.
        side: One of 'top', 'bottom', 'left', 'right'.

    Returns:
        A tuple (x, y) of the wall center coordinates.

    Raises:
        ValueError: If side is not one of the valid options.

    """
    minx, miny, maxx, maxy = room.bounds

    if side == "top":
        return ((minx + maxx) / 2, maxy)
    elif side == "bottom":
        return ((minx + maxx) / 2, miny)
    elif side == "left":
        return (minx, (miny + maxy) / 2)
    elif side == "right":
        return (maxx, (miny + maxy) / 2)
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'top', 'bottom', 'left', or 'right'.")


def create_wall_ring(
    interior: Polygon,
    wall_thickness: float,
) -> Polygon:
    """
    Create a wall ring around an interior space.

    Args:
        interior: The interior space polygon.
        wall_thickness: Thickness of the walls.

    Returns:
        A Shapely Polygon representing the walls (exterior minus interior).

    """
    exterior = interior.buffer(wall_thickness, join_style="mitre")
    return exterior.difference(interior)


def turn_direction(current: Direction, turn: Literal["left", "right"]) -> Direction:
    """
    Get the new direction after turning 90 degrees.

    Args:
        current: Current direction of travel.
        turn: Which way to turn ("left" or "right").

    Returns:
        The new direction after turning.

    """
    if turn == "right":
        # Clockwise: east -> south -> west -> north -> east
        turns = {"east": "south", "south": "west", "west": "north", "north": "east"}
    else:
        # Counterclockwise: east -> north -> west -> south -> east
        turns = {"east": "north", "north": "west", "west": "south", "south": "east"}
    return turns[current]


def move_in_direction(
    start: tuple[float, float],
    direction: Direction,
    distance: float,
) -> tuple[float, float]:
    """
    Calculate the endpoint after moving in a direction.

    Args:
        start: Starting point (x, y).
        direction: Direction to move.
        distance: Distance to move.

    Returns:
        The new point (x, y) after moving.

    """
    x, y = start
    if direction == "east":
        return (x + distance, y)
    elif direction == "west":
        return (x - distance, y)
    elif direction == "north":
        return (x, y + distance)
    else:  # south
        return (x, y - distance)


def create_hallway_segment(
    start: tuple[float, float],
    direction: Direction,
    length: float,
    width: float,
) -> HallwaySegment:
    """
    Create a hallway segment in any direction.

    Args:
        start: Starting point (x, y) of the segment centerline.
        direction: Direction the hallway runs.
        length: Length of the hallway segment.
        width: Width of the hallway.

    Returns:
        A HallwaySegment containing the geometry and metadata.

    """
    end = move_in_direction(start, direction, length)
    half_width = width / 2

    x1, y1 = start
    x2, y2 = end

    if direction in ("east", "west"):
        # Horizontal segment
        min_x, max_x = min(x1, x2), max(x1, x2)
        polygon = box(min_x, y1 - half_width, max_x, y1 + half_width)
    else:
        # Vertical segment
        min_y, max_y = min(y1, y2), max(y1, y2)
        polygon = box(x1 - half_width, min_y, x1 + half_width, max_y)

    return HallwaySegment(start=start, end=end, direction=direction, polygon=polygon)


def opposite_direction(d: Direction) -> Direction:
    """Return the opposite direction."""
    opposites: dict[str, Direction] = {
        "east": "west",
        "west": "east",
        "north": "south",
        "south": "north",
    }
    return opposites[d]


def create_open_space_segment(
    start: tuple[float, float],
    direction: Direction,
    length: float,
    width: float,
    expand_direction: Direction | None = None,
) -> HallwaySegment:
    """
    Create an open space segment — a square polygon offset from the centerline.

    The square's side length equals the segment length. One perpendicular edge
    aligns with the regular hallway edge (width/2 from centerline) so that
    corners connect properly. The opposite edge extends further out so the
    total perpendicular span equals the segment length (making it a square).

    Args:
        start: Starting point (x, y) of the segment centerline.
        direction: Direction the segment runs.
        length: Length of the segment (becomes the square's side length).
        width: Hallway width — used to align the near perpendicular edge.
        expand_direction: Which perpendicular direction gets the large expansion.
            Must be perpendicular to ``direction``. Defaults to "left" of travel.

    Returns:
        A HallwaySegment with is_open_space=True.

    """
    if expand_direction is None:
        expand_direction = get_perpendicular_offset_directions(direction)[0]

    end = move_in_direction(start, direction, length)
    half_width = width / 2
    large_offset = length - half_width

    x1, y1 = start
    x2, y2 = end

    if direction in ("east", "west"):
        min_x, max_x = min(x1, x2), max(x1, x2)
        center_y = y1
        if expand_direction == "north":
            polygon = box(min_x, center_y - half_width, max_x, center_y + large_offset)
        else:
            polygon = box(min_x, center_y - large_offset, max_x, center_y + half_width)
    else:
        min_y, max_y = min(y1, y2), max(y1, y2)
        center_x = x1
        if expand_direction == "east":
            polygon = box(center_x - half_width, min_y, center_x + large_offset, max_y)
        else:
            polygon = box(center_x - large_offset, min_y, center_x + half_width, max_y)

    return HallwaySegment(
        start=start,
        end=end,
        direction=direction,
        polygon=polygon,
        is_open_space=True,
    )


def get_perpendicular_offset_directions(direction: Direction) -> tuple[Direction, Direction]:
    """
    Get the two directions perpendicular to the given direction.

    Args:
        direction: The hallway direction.

    Returns:
        A tuple of (left_side, right_side) directions relative to travel direction.
        For a hallway going east, left is north and right is south.

    """
    perpendicular = {
        "east": ("north", "south"),
        "west": ("south", "north"),
        "north": ("west", "east"),
        "south": ("east", "west"),
    }
    return perpendicular[direction]
