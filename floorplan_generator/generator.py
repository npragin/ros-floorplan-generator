"""Main floorplan generator class."""

import random
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from floorplan_generator.geometry import (
    Direction,
    HallwaySegment,
    create_hallway_segment,
    create_room,
    create_wall_ring,
    get_perpendicular_offset_directions,
    turn_direction,
)
from floorplan_generator.params import FloorplanParams


@dataclass
class Floorplan:
    """
    Represents a complete floorplan layout.

    Attributes:
        hallway_interior: The interior space of the hallway (union of all segments).
        hallway_segments: List of hallway segments (for multi-segment layouts).
        room_interiors: List of room interior polygons.
        doors: List of door opening polygons.
        walls: The combined wall geometry.
        params: Parameters used to generate the floorplan.
        room_ids: List of room IDs corresponding to room_interiors.

    """

    hallway_interior: Polygon
    hallway_segments: list[HallwaySegment]
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

    def __init__(self, params: FloorplanParams, max_retries: int = 10) -> None:
        """
        Initialize the generator with parameters.

        Args:
            params: The floorplan generation parameters.
            max_retries: Maximum retry attempts when collision-free path cannot be found.
                Only applies when seed is not explicitly set.

        Raises:
            ValueError: If parameters are invalid.

        """
        self.params = params
        self.params.validate()
        self.max_retries = max_retries

    def generate(self, debug_dir: str | None = None) -> Floorplan:
        """
        Generate a complete floorplan.

        Args:
            debug_dir: Optional directory to save debug images at each step.

        Returns:
            A Floorplan object containing all geometry.

        """
        # Create hallway segments (handles both straight and multi-turn layouts)
        segments, turn_directions = self._create_hallway_segments()

        # Create corner fill pieces where segments meet
        corner_pieces = self._create_corner_pieces(segments)

        # Combine segments and corners into hallway interior
        hallway_interior = unary_union([seg.polygon for seg in segments] + corner_pieces)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "01_hallway.png",
                hallway=hallway_interior,
            )

        room_interiors, room_ids = self._place_rooms_along_segments(segments, turn_directions)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "02_hallway_and_rooms.png",
                hallway=hallway_interior,
                rooms=room_interiors,
            )

        doors = self._create_doors_for_segments(room_interiors, segments)

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
            hallway_segments=segments,
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

    def _create_hallway_segments(self) -> tuple[list[HallwaySegment], list[str]]:
        """
        Create hallway segments with turns.

        Returns:
            A tuple of (list of HallwaySegment objects, list of turn directions used).
            The turn directions list has length num_turns (one less than segments).

        """
        if self.params.num_turns == 0:
            # Simple case: single straight hallway
            hallway_length = self._calculate_hallway_length()
            hallway_start_x = (
                self.params.room_wall_length - self.params.doorway_width
            ) / 2 - self.params.hallway_end_padding
            return (
                [
                    create_hallway_segment(
                        start=(hallway_start_x, 0),
                        direction="east",
                        length=hallway_length,
                        width=self.params.hallway_width,
                    )
                ],
                [],
            )

        # Multi-segment layout with turns
        num_segments = self.params.num_turns + 1
        rooms_per_segment = self._calculate_rooms_per_segment(num_segments)

        # Pre-compute all turn directions so we know them when calculating segment lengths
        turn_directions_used: list[str] = []
        for turn_idx in range(self.params.num_turns):
            turn_directions_used.append(self._get_next_turn(turn_idx))

        segments: list[HallwaySegment] = []
        # First segment starts at hallway_start_x to align with room doorways
        hallway_start_x = (
            self.params.room_wall_length - self.params.doorway_width
        ) / 2 - self.params.hallway_end_padding
        current_pos = (hallway_start_x, 0.0)
        current_direction: Direction = "east"

        for seg_idx, target_rooms in enumerate(rooms_per_segment):
            # Determine if there are turns at start/end of this segment
            has_turn_at_start = seg_idx > 0
            has_turn_at_end = seg_idx < num_segments - 1

            # Get turn direction at start (for corner skip calculation)
            turn_at_start_dir = turn_directions_used[seg_idx - 1] if has_turn_at_start else None

            # Calculate segment length with turn direction info for proper slot calculation
            segment_length = self._calculate_segment_length(
                target_rooms, has_turn_at_start, has_turn_at_end, turn_at_start_dir
            )

            # Ensure minimum segment length
            segment_length = max(segment_length, self.params.min_segment_length)

            # Create the segment
            segment = create_hallway_segment(
                start=current_pos,
                direction=current_direction,
                length=segment_length,
                width=self.params.hallway_width,
            )
            segments.append(segment)

            # Prepare for next segment (if not last)
            if seg_idx < num_segments - 1:
                # Move to end of current segment
                current_pos = segment.end

                # Apply turn direction
                current_direction = turn_direction(current_direction, turn_directions_used[seg_idx])

        return segments, turn_directions_used

    def _create_corner_pieces(self, segments: list[HallwaySegment]) -> list[Polygon]:
        """
        Create corner fill pieces where hallway segments meet.

        At each turn point, there's a gap where the two perpendicular segments
        don't overlap. This method creates square pieces to fill those gaps.

        Args:
            segments: List of hallway segments.

        Returns:
            List of corner fill polygons.

        """
        if len(segments) <= 1:
            return []

        corners: list[Polygon] = []
        half_width = self.params.hallway_width / 2

        for i in range(len(segments) - 1):
            # The turn point is the end of segment i (same as start of segment i+1)
            turn_point = segments[i].end

            # Create a square centered at the turn point
            corner = box(
                turn_point[0] - half_width,
                turn_point[1] - half_width,
                turn_point[0] + half_width,
                turn_point[1] + half_width,
            )
            corners.append(corner)

        return corners

    def _get_next_turn(self, turn_index: int) -> str:
        """
        Determine the next turn direction based on parameters.

        Args:
            turn_index: Index of the current turn (0-indexed).

        Returns:
            "left" or "right".

        """
        if self.params.turn_direction == "clockwise":
            return "right"
        elif self.params.turn_direction == "counterclockwise":
            return "left"
        elif self.params.turn_direction == "random":
            return random.choice(["left", "right"])
        else:  # alternating
            return "left" if turn_index % 2 == 0 else "right"

    def _calculate_rooms_per_segment(self, num_segments: int) -> list[int]:
        """
        Distribute rooms evenly across segments (differ by at most 1).

        Args:
            num_segments: Number of hallway segments.

        Returns:
            List of room counts for each segment.

        """
        base_rooms = self.params.num_rooms // num_segments
        remainder = self.params.num_rooms % num_segments

        # Spread remainder across segments (first segments get extra)
        return [base_rooms + (1 if i < remainder else 0) for i in range(num_segments)]

    def _calculate_slots_for_segment(
        self,
        target_rooms: int,
        has_turn_at_start: bool,
        turn_at_start_dir: str | None,
    ) -> int:
        """
        Calculate how many room slots are needed to place target_rooms.

        Accounts for corner skip: if there's a turn at start, one room at slot 0
        will be skipped (inside corner), so we need an extra slot.

        Args:
            target_rooms: Number of rooms we want to place on this segment.
            has_turn_at_start: Whether there's a turn at the start of this segment.
            turn_at_start_dir: Direction of the turn at start ("left" or "right").

        Returns:
            Number of slots needed (rooms alternate left/right per slot).

        """
        if target_rooms == 0:
            return 0

        if not has_turn_at_start or turn_at_start_dir is None:
            # No corner skip, slots = ceil(target_rooms / 2) since 2 rooms per slot
            return (target_rooms + 1) // 2

        # One room will be skipped at slot 0 (inside corner)
        # So we need one extra iteration to compensate
        return (target_rooms + 1 + 1) // 2

    def _calculate_segment_length(
        self,
        target_rooms: int,
        has_turn_at_start: bool = False,
        has_turn_at_end: bool = False,
        turn_at_start_dir: str | None = None,
    ) -> float:
        """
        Calculate the required segment length for a given number of rooms.

        Args:
            target_rooms: Number of rooms to place on this segment.
            has_turn_at_start: Whether there's a turn at the start of this segment.
            has_turn_at_end: Whether there's a turn at the end of this segment.
            turn_at_start_dir: Direction of the turn at start ("left" or "right").

        Returns:
            The segment length in meters.

        """
        if target_rooms == 0:
            return self.params.min_segment_length

        # Calculate turn buffer (space needed around corners)
        corridor_gap = max(self.params.wall_thickness, self.params.effective_doorway_length)

        # Calculate slots needed accounting for corner skip
        slots_needed = self._calculate_slots_for_segment(target_rooms, has_turn_at_start, turn_at_start_dir)

        # Effective spacing between rooms on the same side
        effective_spacing = max(self.params.room_spacing, self.params.wall_thickness)

        # Length needed for rooms: slots_needed rooms on one side
        length = slots_needed * self.params.room_wall_length
        if slots_needed > 1:
            length += (slots_needed - 1) * effective_spacing

        # Add turn adjustments only (dead ends need no adjustment since
        # hallway_end_padding - doorway_overhang = 0)
        # For turns: add (turn_buffer - room_wall_length/2) = corridor_gap + hallway_width/2
        turn_adjustment = corridor_gap + self.params.hallway_width / 2

        if has_turn_at_start:
            length += turn_adjustment

        if has_turn_at_end:
            length += turn_adjustment

        return max(length, self.params.min_segment_length)

    def _place_rooms_along_segments(
        self, segments: list[HallwaySegment], turn_directions: list[str]
    ) -> tuple[list[Polygon], list[int]]:
        """
        Place rooms along hallway segments.

        Args:
            segments: List of hallway segments.
            turn_directions: List of turn directions used between segments ("left" or "right").
                Length is len(segments) - 1.

        Returns:
            A tuple of (list of room interior polygons, list of room IDs).

        """
        rooms: list[Polygon] = []
        room_ids: list[int] = []
        room_id_counter = 0

        # Calculate even room distribution per segment
        rooms_per_segment = self._calculate_rooms_per_segment(len(segments))

        for seg_idx, segment in enumerate(segments):
            target_rooms_in_segment = rooms_per_segment[seg_idx]
            if target_rooms_in_segment == 0:
                continue

            # Determine if there's a turn at the start of this segment
            has_turn_at_start = seg_idx > 0

            # Get turn direction for inside corner detection at start
            turn_at_start_dir = turn_directions[seg_idx - 1] if has_turn_at_start else None

            segment_rooms, segment_ids = self._place_rooms_on_segment(
                segment,
                target_rooms_in_segment,
                room_id_counter,
                has_turn_at_start=has_turn_at_start,
                turn_at_start_dir=turn_at_start_dir,
            )
            rooms.extend(segment_rooms)
            room_ids.extend(segment_ids)
            # Track actual rooms placed for ID continuity
            room_id_counter += len(segment_rooms)

        return rooms, room_ids

    def _place_rooms_on_segment(
        self,
        segment: HallwaySegment,
        target_rooms: int,
        start_room_id: int,
        has_turn_at_start: bool = False,
        turn_at_start_dir: str | None = None,
    ) -> tuple[list[Polygon], list[int]]:
        """
        Place rooms along a single hallway segment.

        The preceding segment owns the elbow at each turn, so rooms can extend
        to the segment's end without restriction. Only the following segment
        (at its start) needs to skip the inside corner room.

        Args:
            segment: The hallway segment.
            target_rooms: Number of rooms to place on this segment.
            start_room_id: Starting room ID for numbering.
            has_turn_at_start: Whether there's a turn at the start of this segment.
            turn_at_start_dir: Direction of the turn at start ("left" or "right").

        Returns:
            A tuple of (list of room polygons, list of room IDs).

        """
        rooms: list[Polygon] = []
        room_ids: list[int] = []

        if target_rooms == 0:
            return rooms, room_ids

        # Get perpendicular directions for room placement
        left_dir, right_dir = get_perpendicular_offset_directions(segment.direction)

        # Calculate corridor gap (between hallway edge and room)
        corridor_gap = max(self.params.wall_thickness, self.params.effective_doorway_length)

        # Distance from hallway centerline to room center
        room_offset = self.params.hallway_width / 2 + corridor_gap + self.params.room_wall_length / 2

        # Effective spacing between rooms on the same side
        effective_spacing = max(self.params.room_spacing, self.params.wall_thickness)

        # Turn buffer: don't place rooms too close to turn points
        # Buffer needs to account for room size plus corridor gap
        turn_buffer = self.params.room_wall_length / 2 + corridor_gap + self.params.hallway_width / 2

        # Calculate starting position along the segment
        # For first segment (no turn at start): start at doorway_width/2 + hallway_end_padding
        # For subsequent segments (turn at start): start at turn_buffer from segment start
        if has_turn_at_start:
            room_start_offset = turn_buffer
        else:
            room_start_offset = self.params.doorway_width / 2 + self.params.hallway_end_padding

        if segment.is_horizontal:
            start_along = segment.start[0] + room_start_offset
            if segment.direction == "west":
                start_along = segment.start[0] - room_start_offset
        else:
            start_along = segment.start[1] + room_start_offset
            if segment.direction == "south":
                start_along = segment.start[1] - room_start_offset

        # Place rooms alternating between left and right sides until we have target_rooms
        room_id_counter = start_room_id
        i = 0  # iteration counter for slot/side calculation

        while len(rooms) < target_rooms:
            slot = i // 2  # Which slot along the segment
            is_left_side = i % 2 == 0

            # Skip rooms on the inside of corners at segment START only.
            # After a left turn, the left side is the "inside" of the corner
            # After a right turn, the right side is the "inside" of the corner
            if has_turn_at_start and slot == 0:
                if turn_at_start_dir == "left" and is_left_side:
                    i += 1
                    continue
                if turn_at_start_dir == "right" and not is_left_side:
                    i += 1
                    continue

            # Calculate position along segment
            along_offset = slot * (self.params.room_wall_length + effective_spacing)
            if segment.is_horizontal:
                along_pos = start_along - along_offset if segment.direction == "west" else start_along + along_offset
                center_along = along_pos
                center_perp = segment.start[1]  # Y stays on centerline initially
            else:
                along_pos = start_along - along_offset if segment.direction == "south" else start_along + along_offset
                center_along = segment.start[0]  # X stays on centerline initially
                center_perp = along_pos

            # Determine which side (even=left, odd=right relative to travel direction)
            perp_direction = left_dir if is_left_side else right_dir

            # Calculate room center position
            if segment.is_horizontal:
                room_x = center_along
                room_y = center_perp + room_offset if perp_direction == "north" else center_perp - room_offset
            else:
                room_y = center_perp
                room_x = center_along + room_offset if perp_direction == "east" else center_along - room_offset

            room = create_room(room_x, room_y, self.params.room_wall_length)
            rooms.append(room)
            room_ids.append(room_id_counter)
            room_id_counter += 1
            i += 1

        return rooms, room_ids

    def _create_doors_for_segments(self, rooms: list[Polygon], segments: list[HallwaySegment]) -> list[Polygon]:
        """
        Create door openings between rooms and hallway segments.

        Args:
            rooms: List of room polygons.
            segments: List of hallway segments.

        Returns:
            List of door opening polygons.

        """
        doors: list[Polygon] = []

        for room in rooms:
            room_minx, room_miny, room_maxx, room_maxy = room.bounds
            room_center_x = (room_minx + room_maxx) / 2
            room_center_y = (room_miny + room_maxy) / 2

            # Find which segment this room is adjacent to
            segment = self._find_adjacent_segment(room, segments)

            if segment.is_horizontal:
                # Room is above or below a horizontal hallway
                hallway_miny = segment.polygon.bounds[1]
                hallway_maxy = segment.polygon.bounds[3]

                if room_miny > hallway_maxy:
                    # Room is above hallway
                    door_y_min = hallway_maxy
                    door_y_max = room_miny
                else:
                    # Room is below hallway
                    door_y_min = room_maxy
                    door_y_max = hallway_miny

                door = box(
                    room_center_x - self.params.doorway_width / 2,
                    door_y_min,
                    room_center_x + self.params.doorway_width / 2,
                    door_y_max,
                )
            else:
                # Room is left or right of a vertical hallway
                hallway_minx = segment.polygon.bounds[0]
                hallway_maxx = segment.polygon.bounds[2]

                if room_minx > hallway_maxx:
                    # Room is to the right of hallway
                    door_x_min = hallway_maxx
                    door_x_max = room_minx
                else:
                    # Room is to the left of hallway
                    door_x_min = room_maxx
                    door_x_max = hallway_minx

                door = box(
                    door_x_min,
                    room_center_y - self.params.doorway_width / 2,
                    door_x_max,
                    room_center_y + self.params.doorway_width / 2,
                )

            doors.append(door)

        return doors

    def _find_adjacent_segment(self, room: Polygon, segments: list[HallwaySegment]) -> HallwaySegment:
        """
        Find which hallway segment a room is adjacent to.

        Args:
            room: The room polygon.
            segments: List of hallway segments.

        Returns:
            The adjacent HallwaySegment.

        """
        room_center_x = (room.bounds[0] + room.bounds[2]) / 2
        room_center_y = (room.bounds[1] + room.bounds[3]) / 2

        best_segment = segments[0]
        best_distance = float("inf")

        for segment in segments:
            # Calculate distance from room center to segment centerline
            if segment.is_horizontal:
                # For horizontal segments, check if room x is within segment x range
                seg_minx = min(segment.start[0], segment.end[0])
                seg_maxx = max(segment.start[0], segment.end[0])
                if seg_minx <= room_center_x <= seg_maxx:
                    distance = abs(room_center_y - segment.start[1])
                    if distance < best_distance:
                        best_distance = distance
                        best_segment = segment
            else:
                # For vertical segments, check if room y is within segment y range
                seg_miny = min(segment.start[1], segment.end[1])
                seg_maxy = max(segment.start[1], segment.end[1])
                if seg_miny <= room_center_y <= seg_maxy:
                    distance = abs(room_center_x - segment.start[0])
                    if distance < best_distance:
                        best_distance = distance
                        best_segment = segment

        return best_segment

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
