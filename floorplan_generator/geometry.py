"""Geometric helper functions using Shapely."""

from shapely.geometry import Polygon, box


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
