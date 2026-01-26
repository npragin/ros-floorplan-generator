"""Main floorplan generator class."""

from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorplan_generator.geometry import (
    create_hallway,
    create_room,
    create_wall_ring,
)
from floorplan_generator.params import FloorplanParams


@dataclass
class Floorplan:
    """
    Represents a complete floorplan layout.

    Attributes:
        hallway_interior: The interior space of the hallway.
        room_interiors: List of room interior polygons.
        doors: List of door opening polygons.
        walls: The combined wall geometry.
        params: Parameters used to generate the floorplan.

    """

    hallway_interior: Polygon
    room_interiors: list[Polygon]
    doors: list[Polygon]
    walls: Polygon
    params: FloorplanParams
    room_ids: list[int] = field(default_factory=list)

    def get_bounds(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the entire floorplan.

        Returns:
            A tuple (min_x, min_y, max_x, max_y).

        """
        return self.walls.bounds

    def get_free_space(self) -> Polygon:
        """
        Return the combined free space (rooms + hallway + doors).

        Returns:
            A Shapely Polygon representing all navigable space.

        """
        all_interiors = [self.hallway_interior, *self.room_interiors, *self.doors]
        return unary_union(all_interiors)


class FloorplanGenerator:
    """Generates office-style floorplans with a linear hallway layout."""

    def __init__(self, params: FloorplanParams) -> None:
        """
        Initialize the generator with parameters.

        Args:
            params: The floorplan generation parameters.

        Raises:
            ValueError: If parameters are invalid.

        """
        self.params = params
        self.params.validate()

    def generate(self, debug_dir: str | None = None) -> Floorplan:
        """
        Generate a complete floorplan.

        Args:
            debug_dir: Optional directory to save debug images at each step.

        Returns:
            A Floorplan object containing all geometry.

        """
        hallway_length = self._calculate_hallway_length()
        hallway_interior = self._create_hallway(hallway_length)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "01_hallway.png",
                hallway=hallway_interior,
            )

        room_interiors, room_ids = self._place_rooms(hallway_length)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "02_hallway_and_rooms.png",
                hallway=hallway_interior,
                rooms=room_interiors,
            )

        doors = self._create_doors(room_interiors, hallway_interior)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "03_with_doors.png",
                hallway=hallway_interior,
                rooms=room_interiors,
                doors=doors,
            )

        walls = self._create_walls(hallway_interior, room_interiors, doors)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "04_with_walls.png",
                hallway=hallway_interior,
                rooms=room_interiors,
                doors=doors,
                walls=walls,
            )

        return Floorplan(
            hallway_interior=hallway_interior,
            room_interiors=room_interiors,
            doors=doors,
            walls=walls,
            params=self.params,
            room_ids=room_ids,
        )

    def _render_step(
        self,
        output_path: Path,
        hallway: Polygon | None = None,
        rooms: list[Polygon] | None = None,
        doors: list[Polygon] | None = None,
        walls: Polygon | None = None,
    ) -> None:
        """
        Render a debug image showing the current generation step.

        Args:
            output_path: Path to save the debug image.
            hallway: The hallway interior polygon.
            rooms: List of room interior polygons.
            doors: List of door polygons.
            walls: The walls polygon.

        """
        from pathlib import Path

        import numpy as np
        from PIL import Image, ImageDraw
        from shapely.geometry import MultiPolygon

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate bounds from all geometry
        all_geoms = []
        if hallway:
            all_geoms.append(hallway)
        if rooms:
            all_geoms.extend(rooms)
        if doors:
            all_geoms.extend(doors)
        if walls:
            all_geoms.append(walls)

        if not all_geoms:
            return

        combined = unary_union(all_geoms)
        min_x, min_y, max_x, max_y = combined.bounds
        padding = 1.0
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        resolution = 0.05
        width_px = int(np.ceil((max_x - min_x) / resolution))
        height_px = int(np.ceil((max_y - min_y) / resolution))

        image = Image.new("RGB", (width_px, height_px), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)

        def world_to_pixel(x: float, y: float) -> tuple[int, int]:
            px = round((x - min_x) / resolution)
            py = height_px - 1 - round((y - min_y) / resolution)
            return (px, py)

        def ring_to_pixel_coords(ring) -> list[tuple[int, int]]:
            """Convert a LinearRing to pixel coordinates."""
            return [world_to_pixel(x, y) for x, y in ring.coords]

        def draw_polygon_with_holes(
            polygon: Polygon, fill: tuple[int, int, int], hole_fill: tuple[int, int, int]
        ) -> None:
            """Draw a polygon, properly handling interior holes."""
            exterior_coords = ring_to_pixel_coords(polygon.exterior)
            draw.polygon(exterior_coords, fill=fill)
            for interior in polygon.interiors:
                hole_coords = ring_to_pixel_coords(interior)
                draw.polygon(hole_coords, fill=hole_fill)

        def draw_polygon(polygon: Polygon, fill: tuple[int, int, int]) -> None:
            """Draw a simple polygon (no hole handling needed for rooms/hallway/doors)."""
            if isinstance(polygon, MultiPolygon):
                for geom in polygon.geoms:
                    if isinstance(geom, Polygon):
                        exterior_coords = ring_to_pixel_coords(geom.exterior)
                        draw.polygon(exterior_coords, fill=fill)
            elif isinstance(polygon, Polygon):
                exterior_coords = ring_to_pixel_coords(polygon.exterior)
                draw.polygon(exterior_coords, fill=fill)

        bg_color = (240, 240, 240)

        # Draw walls first (dark gray), with holes as background color
        if walls:
            if isinstance(walls, MultiPolygon):
                for geom in walls.geoms:
                    if isinstance(geom, Polygon):
                        draw_polygon_with_holes(geom, (60, 60, 60), bg_color)
            elif isinstance(walls, Polygon):
                draw_polygon_with_holes(walls, (60, 60, 60), bg_color)

        # Draw hallway interior (light blue)
        if hallway:
            draw_polygon(hallway, (200, 220, 255))

        # Draw room interiors (light green)
        if rooms:
            for room in rooms:
                draw_polygon(room, (200, 255, 200))

        # Draw doors (orange)
        if doors:
            for door in doors:
                draw_polygon(door, (255, 165, 0))

        image.save(output_path)
        print(f"  Debug: saved {output_path.name}")

    def _calculate_hallway_length(self) -> float:
        """
        Calculate the required hallway length.

        Returns:
            The hallway length in meters.

        """
        # For alternating rooms on both sides:
        # - Rooms on each side need: room_wall_length + effective_spacing
        # - Number of rooms on each side: ceil(num_rooms / 2) and floor(num_rooms / 2)
        # - The hallway length is determined by the side with more rooms

        rooms_top = (self.params.num_rooms + 1) // 2  # Ceiling division
        rooms_per_side = max(rooms_top, self.params.num_rooms - rooms_top)

        # Effective spacing must account for a shared wall between adjacent rooms
        # Even with room_spacing=0, we need wall_thickness for the shared wall
        effective_spacing = max(self.params.room_spacing, self.params.wall_thickness)

        # Each room takes up room_wall_length along the hallway
        # Plus spacing between rooms (but not after the last one)
        length = rooms_per_side * self.params.room_wall_length
        if rooms_per_side > 1:
            length += (rooms_per_side - 1) * effective_spacing

        # Subtract the space taken by one room beyond the doorway so that the hallway ends at the
        # doorway's end with 0 padding. The difference between room edge and doorway edge is
        # (room_wall_length - doorway_width) / 2 on each side.
        length -= self.params.room_wall_length - self.params.doorway_width

        # Add padding at the ends (on each side)
        length += 2 * self.params.hallway_end_padding

        return length

    def _create_hallway(self, hallway_length: float) -> Polygon:
        """
        Create the main hallway.

        Args:
            hallway_length: Length of the hallway.

        Returns:
            A Polygon representing the hallway interior.

        """
        # Calculate the hallway start position to align with the first doorway's left edge when padding=0,
        # and extend padding distance before it when padding > 0.
        # First doorway left edge (when padding=0) is at: (room_wall_length/2) - doorway_width/2
        # = (room_wall_length - doorway_width) / 2
        # With padding, we subtract it to extend the hallway before the first doorway.
        hallway_start_x = (
            self.params.room_wall_length - self.params.doorway_width
        ) / 2 - self.params.hallway_end_padding

        return create_hallway(
            start_x=hallway_start_x,
            start_y=0,
            length=hallway_length,
            width=self.params.hallway_width,
        )

    def _place_rooms(self, hallway_length: float) -> tuple[list[Polygon], list[int]]:
        """
        Place rooms along both sides of the hallway.

        Args:
            hallway_length: Length of the hallway.

        Returns:
            A tuple of (list of room interior polygons, list of room IDs).

        """
        rooms: list[Polygon] = []
        room_ids: list[int] = []

        # Calculate the gap between hallway and room (for the doorway corridor)
        # The gap needs to be at least wall_thickness, but larger for extended doorways
        # Since door is centered in gap, gap should accommodate half the door length on each side
        corridor_gap = max(self.params.wall_thickness, self.params.effective_doorway_length)

        # Calculate the y-offset for rooms (from hallway center to room center)
        # Room center is at: hallway_width/2 + corridor_gap + room_wall_length/2
        y_offset = self.params.hallway_width / 2 + corridor_gap + self.params.room_wall_length / 2

        # Effective spacing must account for a shared wall between adjacent rooms
        # Even with room_spacing=0, we need wall_thickness for the shared wall
        effective_spacing = max(self.params.room_spacing, self.params.wall_thickness)

        # Starting x position: half room width (to center the first room at room_wall_length/2)
        # Padding is handled by the hallway, not by shifting room positions
        start_x = self.params.room_wall_length / 2

        # Place rooms alternating between top and bottom
        for i in range(self.params.num_rooms):
            # Calculate which "slot" this room is in (0-indexed)
            slot = i // 2

            # Calculate x position
            x = start_x + slot * (self.params.room_wall_length + effective_spacing)

            # Determine if room is on top (even index) or bottom (odd index)
            y = y_offset if i % 2 == 0 else -y_offset

            room = create_room(x, y, self.params.room_wall_length)
            rooms.append(room)
            room_ids.append(i)

        return rooms, room_ids

    def _create_doors(
        self,
        rooms: list[Polygon],
        hallway: Polygon,
    ) -> list[Polygon]:
        """
        Create door openings between rooms and hallway.

        Args:
            rooms: List of room polygons.
            hallway: The hallway polygon.

        Returns:
            List of door opening polygons.

        """
        doors: list[Polygon] = []
        hallway_miny, hallway_maxy = hallway.bounds[1], hallway.bounds[3]

        for room in rooms:
            # Determine which side the door should be on
            room_minx, room_miny, room_maxx, room_maxy = room.bounds

            if room_miny > hallway_maxy:
                # Room is above hallway, door connects room bottom to hallway top
                # Door must span from inside hallway to inside room to ensure connection
                door_y_min = hallway_maxy
                door_y_max = room_miny
            else:
                # Room is below hallway, door connects room top to hallway bottom
                door_y_min = room_maxy
                door_y_max = hallway_miny

            # X position is centered on the room
            door_x = (room_minx + room_maxx) / 2

            # Create door opening using doorway_width
            from shapely.geometry import box

            door = box(
                door_x - self.params.doorway_width / 2,
                door_y_min,
                door_x + self.params.doorway_width / 2,
                door_y_max,
            )
            doors.append(door)

        return doors

    def _create_walls(
        self,
        hallway_interior: Polygon,
        room_interiors: list[Polygon],
        doors: list[Polygon],
    ) -> Polygon:
        """
        Create the wall geometry.

        Args:
            hallway_interior: The hallway interior space.
            room_interiors: List of room interior spaces.
            doors: List of door openings.

        Returns:
            A Polygon representing all walls.

        """
        # Create the combined free space (rooms + hallway + door corridors)
        # This represents all navigable area as a single unified geometry
        free_space = unary_union([hallway_interior, *room_interiors, *doors])

        # Create a single wall ring around the entire free space
        # This naturally handles shared walls between adjacent rooms, doors, and connections
        all_walls = create_wall_ring(free_space, self.params.wall_thickness)

        return all_walls
