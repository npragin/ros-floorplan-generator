"""Main floorplan generator class."""

import random
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from floorplan_generator.collision import CollisionError, plan_path
from floorplan_generator.geometry import (
    Direction,
    HallwaySegment,
    create_hallway_segment,
    create_open_space_segment,
    create_room,
    create_wall_ring,
    get_perpendicular_offset_directions,
    opposite_direction,
)
from floorplan_generator.geometry import (
    turn_direction as turn_direction_fn,
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

        Raises:
            ValueError: If a collision-free hallway path cannot be found.

        """
        # Create hallway segments with collision detection and retry logic
        segments, turn_directions, open_space_indices = self._create_hallway_segments_with_retries()

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "00_segments.png",
                hallway=unary_union([seg.polygon for seg in segments]),
            )

        # Create corner fill pieces where segments meet
        corner_pieces = self._create_corner_pieces(segments)

        # Combine segments and corners into hallway interior
        hallway_interior = unary_union([seg.polygon for seg in segments] + corner_pieces)

        if debug_dir:
            self._render_step(
                Path(debug_dir) / "01_hallway.png",
                hallway=hallway_interior,
            )

        room_interiors, room_ids = self._place_rooms_along_segments(segments, turn_directions, open_space_indices)

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

    def _create_hallway_segments_with_retries(
        self,
    ) -> tuple[list[HallwaySegment], list[str], set[int]]:
        """
        Create hallway segments, retrying with new seeds on collision.

        Seeding is expected to be done by the caller (main.py) before invoking
        generate(). This method only re-seeds on retry with system entropy.

        Returns:
            A tuple of (list of HallwaySegment objects, list of turn directions used,
            set of open space segment indices).

        Raises:
            ValueError: If no collision-free path can be found after all retries.

        """
        # Deterministic turn modes exhaust all options via backtracking, no retry needed
        deterministic = self.params.turn_direction in ("alternating", "clockwise", "counterclockwise")

        if deterministic or self.max_retries == 0:
            try:
                return self._create_hallway_segments()
            except CollisionError as e:
                raise ValueError(f"Cannot generate collision-free hallway layout: {e}") from None

        for attempt in range(self.max_retries + 1):
            try:
                return self._create_hallway_segments()
            except CollisionError:
                if attempt < self.max_retries:
                    random.seed()
                    continue
                raise ValueError(
                    f"Cannot generate collision-free hallway layout after {self.max_retries + 1} attempts. "
                    "Try reducing num_turns or changing turn_direction."
                ) from None

        raise ValueError("Unexpected: exhausted retry loop without result.")  # pragma: no cover

    def _create_hallway_segments(self) -> tuple[list[HallwaySegment], list[str], set[int]]:
        """
        Create hallway segments with turns.

        Returns:
            A tuple of (list of HallwaySegment objects, list of turn directions used,
            set of open space segment indices).
            The turn directions list has length num_turns (one less than segments).

        """
        if self.params.num_turns == 0:
            # Simple case: single straight hallway
            hallway_length = self._calculate_hallway_length()
            hallway_start_x = (
                self.params.room_wall_length - self.params.doorway_width
            ) / 2 - self.params.hallway_end_padding

            open_space_indices: set[int] = set()
            if self.params.num_open_spaces > 0:
                open_space_indices = {0}

            if 0 in open_space_indices:
                segment = create_open_space_segment(
                    start=(hallway_start_x, 0),
                    direction="east",
                    length=hallway_length,
                    width=self.params.hallway_width,
                )
            else:
                segment = create_hallway_segment(
                    start=(hallway_start_x, 0),
                    direction="east",
                    length=hallway_length,
                    width=self.params.hallway_width,
                )
            return [segment], [], open_space_indices

        # Multi-segment layout with turns
        num_segments = self.params.num_turns + 1

        # Randomly select which segments become open spaces
        open_space_indices = set()
        if self.params.num_open_spaces > 0:
            open_space_indices = set(random.sample(range(num_segments), self.params.num_open_spaces))

        rooms_per_segment = self._calculate_rooms_per_segment(num_segments, open_space_indices)

        # First segment starts at hallway_start_x to align with room doorways
        hallway_start_x = (
            self.params.room_wall_length - self.params.doorway_width
        ) / 2 - self.params.hallway_end_padding
        start_pos = (hallway_start_x, 0.0)

        # Build a segment length function that captures rooms_per_segment context
        def segment_length_fn(seg_idx: int, turn_at_start_dir: str | None, direction: Direction) -> float:
            target_rooms = rooms_per_segment[seg_idx]

            if seg_idx in open_space_indices:
                num_available_sides = self._get_num_available_sides(seg_idx, num_segments)

                # Determine connection sides from directions
                connection_sides: list[Direction] = []
                if seg_idx > 0 and turn_at_start_dir is not None:
                    # Previous direction = reverse the turn from current direction
                    reverse_turn = "right" if turn_at_start_dir == "left" else "left"
                    prev_direction = turn_direction_fn(direction, reverse_turn)
                    connection_sides.append(opposite_direction(prev_direction))
                if seg_idx < num_segments - 1:
                    connection_sides.append(direction)

                length = self._calculate_open_space_length(target_rooms, num_available_sides, connection_sides)
            else:
                has_turn_at_start = seg_idx > 0
                has_turn_at_end = seg_idx < num_segments - 1
                length = self._calculate_segment_length(
                    target_rooms, has_turn_at_start, has_turn_at_end, turn_at_start_dir
                )

            return max(length, self.params.min_segment_length)

        # Use plan_path with backtracking for collision-free layout
        planned_segments, turn_directions_used = plan_path(
            start_pos=start_pos,
            start_direction="east",
            num_segments=num_segments,
            segment_length_fn=segment_length_fn,
            turn_chooser=self._get_next_turn,
            hallway_width=self.params.hallway_width,
            open_space_indices=open_space_indices,
        )

        # Convert PlannedSegments to HallwaySegments
        segments: list[HallwaySegment] = []
        for planned in planned_segments:
            if planned.index in open_space_indices:
                segment = create_open_space_segment(
                    start=planned.start,
                    direction=planned.direction,
                    length=planned.length,
                    width=self.params.hallway_width,
                    expand_direction=planned.expand_direction,
                )
            else:
                segment = create_hallway_segment(
                    start=planned.start,
                    direction=planned.direction,
                    length=planned.length,
                    width=self.params.hallway_width,
                )
            segments.append(segment)

        return segments, turn_directions_used, open_space_indices

    def _create_corner_pieces(self, segments: list[HallwaySegment]) -> list[Polygon]:
        """
        Create corner fill pieces where hallway segments meet.

        At each turn point, there's a gap where the two perpendicular segments
        don't overlap. This method creates pieces to fill those gaps. When a
        segment is an open space, the corner spans the full perpendicular
        extent of the open space polygon.

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
            seg_i = segments[i]
            seg_j = segments[i + 1]

            if not seg_i.is_open_space and not seg_j.is_open_space:
                # Both regular: standard hw x hw square
                turn_point = seg_i.end
                corner = box(
                    turn_point[0] - half_width,
                    turn_point[1] - half_width,
                    turn_point[0] + half_width,
                    turn_point[1] + half_width,
                )
            else:
                # At least one open space: corner spans both segments'
                # perpendicular extents so the junction is fully connected.
                bi = seg_i.polygon.bounds
                bj = seg_j.polygon.bounds
                # Use each segment's perpendicular extent for the corner
                corner = box(bj[0], bi[1], bj[2], bi[3]) if seg_i.is_horizontal else box(bi[0], bj[1], bi[2], bj[3])

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

    def _calculate_rooms_per_segment(
        self,
        num_segments: int,
        open_space_indices: set[int] | None = None,
    ) -> list[int]:
        """
        Distribute rooms across segments proportionally to capacity.

        Regular segments have 2 sides. Open spaces have 2-4 sides depending
        on position (connections reduce available sides). Rooms are distributed
        proportionally to each segment's side count.

        Args:
            num_segments: Number of hallway segments.
            open_space_indices: Set of segment indices that are open spaces.

        Returns:
            List of room counts for each segment.

        """
        if open_space_indices is None:
            open_space_indices = set()

        # Calculate weight (available sides) per segment
        weights = []
        for i in range(num_segments):
            if i in open_space_indices:
                weights.append(self._get_num_available_sides(i, num_segments))
            else:
                weights.append(2)

        total_weight = sum(weights)
        total_rooms = self.params.num_rooms

        # Distribute proportionally with integer rounding
        result = [0] * num_segments
        assigned = 0
        for i in range(num_segments):
            result[i] = int(total_rooms * weights[i] / total_weight)
            assigned += result[i]

        # Distribute remainder to highest-weight segments first
        remainder = total_rooms - assigned
        # Sort indices by weight descending, then by index for stability
        indices_by_weight = sorted(range(num_segments), key=lambda i: (-weights[i], i))
        for idx in indices_by_weight:
            if remainder <= 0:
                break
            result[idx] += 1
            remainder -= 1

        return result

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

    def _detect_loop_end_skips(self, segments: list[HallwaySegment]) -> dict[int, str]:
        """
        Detect where a segment's end approaches a non-adjacent segment (loop closure).

        When hallway segments form a loop, the closing segment's end may be close
        to an earlier non-adjacent segment. Rooms at that end on the inside of the
        virtual corner would overlap with rooms from the approached segment.

        Returns:
            A dict mapping segment index -> virtual turn direction ("left" or "right")
            for segments that need an end-skip.

        """
        corridor_gap = max(self.params.wall_thickness, self.params.effective_doorway_length)
        proximity_threshold = self.params.hallway_width / 2 + corridor_gap + self.params.room_wall_length

        end_skips: dict[int, str] = {}

        for i, seg_i in enumerate(segments):
            for j, seg_j in enumerate(segments):
                if abs(i - j) <= 1:
                    continue

                distance = Point(seg_i.end).distance(seg_j.polygon)
                if distance >= proximity_threshold:
                    continue

                # Determine which side of seg_i the approached segment lies on
                left_dir, _right_dir = get_perpendicular_offset_directions(seg_i.direction)

                # Project the approached segment's midpoint relative to seg_i's end
                mid_j = ((seg_j.start[0] + seg_j.end[0]) / 2, (seg_j.start[1] + seg_j.end[1]) / 2)
                dx = mid_j[0] - seg_i.end[0]
                dy = mid_j[1] - seg_i.end[1]

                # Check which perpendicular side the approached segment is on
                if left_dir == "north":
                    is_left = dy > 0
                elif left_dir == "south":
                    is_left = dy < 0
                elif left_dir == "east":
                    is_left = dx > 0
                else:  # west
                    is_left = dx < 0

                # The virtual turn direction is the side where the approached segment is
                end_skips[i] = "left" if is_left else "right"
                break  # Only need one match per segment

        return end_skips

    def _place_rooms_along_segments(
        self,
        segments: list[HallwaySegment],
        turn_directions: list[str],
        open_space_indices: set[int] | None = None,
    ) -> tuple[list[Polygon], list[int]]:
        """
        Place rooms along hallway segments.

        Args:
            segments: List of hallway segments.
            turn_directions: List of turn directions used between segments ("left" or "right").
                Length is len(segments) - 1.
            open_space_indices: Set of segment indices that are open spaces.

        Returns:
            A tuple of (list of room interior polygons, list of room IDs).

        """
        rooms: list[Polygon] = []
        room_ids: list[int] = []
        room_id_counter = 0

        if open_space_indices is None:
            open_space_indices = set()

        # Calculate room distribution per segment (weighted by available sides)
        rooms_per_segment = self._calculate_rooms_per_segment(len(segments), open_space_indices)

        # Detect loop closure points where end-of-segment rooms need skipping
        end_skip_map = self._detect_loop_end_skips(segments)

        for seg_idx, segment in enumerate(segments):
            target_rooms_in_segment = rooms_per_segment[seg_idx]
            if target_rooms_in_segment == 0:
                continue

            if seg_idx in open_space_indices:
                # Open space: place rooms on available sides
                segment_rooms, segment_ids = self._place_rooms_on_open_space(
                    segment,
                    seg_idx,
                    segments,
                    target_rooms_in_segment,
                    room_id_counter,
                )
            else:
                # Regular segment
                has_turn_at_start = seg_idx > 0
                turn_at_start_dir = turn_directions[seg_idx - 1] if has_turn_at_start else None
                has_turn_at_end = seg_idx in end_skip_map
                turn_at_end_dir = end_skip_map.get(seg_idx)

                segment_rooms, segment_ids = self._place_rooms_on_segment(
                    segment,
                    target_rooms_in_segment,
                    room_id_counter,
                    has_turn_at_start=has_turn_at_start,
                    turn_at_start_dir=turn_at_start_dir,
                    has_turn_at_end=has_turn_at_end,
                    turn_at_end_dir=turn_at_end_dir,
                )

            rooms.extend(segment_rooms)
            room_ids.extend(segment_ids)
            room_id_counter += len(segment_rooms)

        return rooms, room_ids

    def _place_rooms_on_segment(
        self,
        segment: HallwaySegment,
        target_rooms: int,
        start_room_id: int,
        has_turn_at_start: bool = False,
        turn_at_start_dir: str | None = None,
        has_turn_at_end: bool = False,
        turn_at_end_dir: str | None = None,
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
            has_turn_at_end: Whether there's a virtual turn at the end (loop closure).
            turn_at_end_dir: Direction of the virtual turn at end ("left" or "right").

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

        # Calculate total slots to determine the last slot index for end-skip
        total_slots = self._calculate_slots_for_segment(
            target_rooms,
            has_turn_at_start,
            turn_at_start_dir,
        )
        last_slot = total_slots - 1

        # Place rooms alternating between left and right sides until we have enough
        rooms_to_place = target_rooms
        room_id_counter = start_room_id
        i = 0  # iteration counter for slot/side calculation

        while len(rooms) < rooms_to_place:
            slot = i // 2  # Which slot along the segment
            is_left_side = i % 2 == 0

            # Skip rooms on the inside of corners at segment START.
            # After a left turn, the left side is the "inside" of the corner
            # After a right turn, the right side is the "inside" of the corner
            if has_turn_at_start and slot == 0:
                if turn_at_start_dir == "left" and is_left_side:
                    i += 1
                    continue
                if turn_at_start_dir == "right" and not is_left_side:
                    i += 1
                    continue

            # Skip rooms on the inside of virtual corners at segment END (loop closure).
            # Decrement rooms_to_place so the loop accepts placing one fewer room.
            if has_turn_at_end and slot == last_slot:
                if turn_at_end_dir == "left" and is_left_side:
                    rooms_to_place -= 1
                    i += 1
                    continue
                if turn_at_end_dir == "right" and not is_left_side:
                    rooms_to_place -= 1
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

    def _get_open_space_available_sides(
        self,
        seg_idx: int,
        segments: list[HallwaySegment],
    ) -> list[Direction]:
        """
        Get the sides of an open space available for room placement.

        Connection sides (entry/exit to adjacent segments) are excluded.

        Args:
            seg_idx: Index of the open space segment.
            segments: All hallway segments.

        Returns:
            List of Direction values where rooms can be placed.

        """
        segment = segments[seg_idx]
        all_sides: set[Direction] = {"east", "west", "north", "south"}
        connection_sides: set[Direction] = set()

        # Entry side: where the previous segment connects
        if seg_idx > 0:
            prev_seg = segments[seg_idx - 1]
            connection_sides.add(opposite_direction(prev_seg.direction))

        # Exit side: where the next segment connects
        if seg_idx < len(segments) - 1:
            connection_sides.add(segment.direction)

        return sorted(all_sides - connection_sides)

    def _get_num_available_sides(self, seg_idx: int, num_segments: int) -> int:
        """
        Get the number of available sides for an open space at a given position.

        Based on connection count: first/last segments have 1 connection,
        intermediate segments have 2, and a single segment has 0.

        Args:
            seg_idx: Index of the segment.
            num_segments: Total number of segments.

        Returns:
            Number of available sides (4, 3, or 2).

        """
        connections = 0
        if seg_idx > 0:
            connections += 1
        if seg_idx < num_segments - 1:
            connections += 1
        return 4 - connections

    def _calculate_open_space_length(
        self,
        target_rooms: int,
        num_available_sides: int,
        connection_sides: list[Direction],
    ) -> float:
        """
        Calculate the required open space side length for a given number of rooms.

        Distributes rooms across available sides and sizes the square to fit
        the busiest side. Adds turn buffer space for each unique axis that has
        a hallway connection, so rooms on perpendicular sides have clearance.

        Args:
            target_rooms: Number of rooms to place on this open space.
            num_available_sides: Number of sides available for room placement.
            connection_sides: Directions where hallway connections exist.

        Returns:
            The open space side length in meters.

        """
        if target_rooms == 0 or num_available_sides == 0:
            return self.params.min_segment_length

        # Distribute rooms across sides (same logic as _place_rooms_on_open_space)
        base_per_side = target_rooms // num_available_sides
        remainder = target_rooms % num_available_sides
        max_rooms_on_side = base_per_side + (1 if remainder > 0 else 0)

        effective_spacing = max(self.params.room_spacing, self.params.wall_thickness)

        length = max_rooms_on_side * self.params.room_wall_length
        if max_rooms_on_side > 1:
            length += (max_rooms_on_side - 1) * effective_spacing

        # Add turn adjustment for each unique axis with a connection.
        # Connections on the same axis (e.g. west+east) only need one adjustment
        # since they constrain the same perpendicular dimension.
        corridor_gap = max(self.params.wall_thickness, self.params.effective_doorway_length)
        turn_adjustment = corridor_gap + self.params.hallway_width / 2

        axes_with_connections: set[str] = set()
        for side in connection_sides:
            if side in ("east", "west"):
                axes_with_connections.add("horizontal")
            else:
                axes_with_connections.add("vertical")
        length += len(axes_with_connections) * turn_adjustment

        print("num axes with connections:", len(axes_with_connections))

        return max(length, self.params.min_segment_length)

    def _place_rooms_on_open_space(
        self,
        segment: HallwaySegment,
        seg_idx: int,
        segments: list[HallwaySegment],
        target_rooms: int,
        start_room_id: int,
    ) -> tuple[list[Polygon], list[int]]:
        """
        Place rooms around an open space segment.

        Rooms are distributed evenly across available sides (non-connection sides).

        Args:
            segment: The open space hallway segment.
            seg_idx: Index of this segment in the segments list.
            segments: All hallway segments.
            target_rooms: Number of rooms to place.
            start_room_id: Starting room ID for numbering.

        Returns:
            A tuple of (list of room polygons, list of room IDs).

        """
        rooms: list[Polygon] = []
        room_ids: list[int] = []

        available_sides = self._get_open_space_available_sides(seg_idx, segments)
        if not available_sides or target_rooms == 0:
            return rooms, room_ids

        # Distribute rooms across available sides
        num_sides = len(available_sides)
        base_per_side = target_rooms // num_sides
        remainder = target_rooms % num_sides
        rooms_on_side = [base_per_side + (1 if i < remainder else 0) for i in range(num_sides)]

        # Calculate corridor gap and room offset from edge
        corridor_gap = max(self.params.wall_thickness, self.params.effective_doorway_length)
        effective_spacing = max(self.params.room_spacing, self.params.wall_thickness)

        # The open space polygon bounds
        minx, miny, maxx, maxy = segment.polygon.bounds
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        room_id_counter = start_room_id

        for side_idx, side_dir in enumerate(available_sides):
            num_rooms_this_side = rooms_on_side[side_idx]
            if num_rooms_this_side == 0:
                continue

            # Room center distance from the square edge
            room_offset = corridor_gap + self.params.room_wall_length / 2

            # Calculate positions along this side
            total_room_span = (
                num_rooms_this_side * self.params.room_wall_length + max(0, num_rooms_this_side - 1) * effective_spacing
            )
            # Center the rooms along the side
            start_offset = -total_room_span / 2 + self.params.room_wall_length / 2

            for room_i in range(num_rooms_this_side):
                along_offset = start_offset + room_i * (self.params.room_wall_length + effective_spacing)

                if side_dir == "north":
                    room_x = center_x + along_offset
                    room_y = maxy + room_offset
                elif side_dir == "south":
                    room_x = center_x + along_offset
                    room_y = miny - room_offset
                elif side_dir == "east":
                    room_x = maxx + room_offset
                    room_y = center_y + along_offset
                else:  # west
                    room_x = minx - room_offset
                    room_y = center_y + along_offset

                room = create_room(room_x, room_y, self.params.room_wall_length)
                rooms.append(room)
                room_ids.append(room_id_counter)
                room_id_counter += 1

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

            seg_minx, seg_miny, seg_maxx, seg_maxy = segment.polygon.bounds

            if segment.is_open_space:
                # For open spaces, determine which side the room is on
                # by comparing room center to the square's bounds
                dx_left = abs(room_center_x - seg_minx)
                dx_right = abs(room_center_x - seg_maxx)
                dy_bottom = abs(room_center_y - seg_miny)
                dy_top = abs(room_center_y - seg_maxy)

                min_dist = min(dx_left, dx_right, dy_bottom, dy_top)

                if min_dist in (dy_top, dy_bottom):
                    # Room is above or below the open space
                    if room_miny > seg_maxy:
                        door_y_min = seg_maxy
                        door_y_max = room_miny
                    else:
                        door_y_min = room_maxy
                        door_y_max = seg_miny
                    door = box(
                        room_center_x - self.params.doorway_width / 2,
                        door_y_min,
                        room_center_x + self.params.doorway_width / 2,
                        door_y_max,
                    )
                else:
                    # Room is left or right of the open space
                    if room_minx > seg_maxx:
                        door_x_min = seg_maxx
                        door_x_max = room_minx
                    else:
                        door_x_min = room_maxx
                        door_x_max = seg_minx
                    door = box(
                        door_x_min,
                        room_center_y - self.params.doorway_width / 2,
                        door_x_max,
                        room_center_y + self.params.doorway_width / 2,
                    )
            elif segment.is_horizontal:
                # Room is above or below a horizontal hallway
                if room_miny > seg_maxy:
                    door_y_min = seg_maxy
                    door_y_max = room_miny
                else:
                    door_y_min = room_maxy
                    door_y_max = seg_miny

                door = box(
                    room_center_x - self.params.doorway_width / 2,
                    door_y_min,
                    room_center_x + self.params.doorway_width / 2,
                    door_y_max,
                )
            else:
                # Room is left or right of a vertical hallway
                if room_minx > seg_maxx:
                    door_x_min = seg_maxx
                    door_x_max = room_minx
                else:
                    door_x_min = room_maxx
                    door_x_max = seg_minx

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

        For open space segments, rooms can be on any of 4 sides, so both axes
        are checked against the square's bounds.

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
            if segment.is_open_space:
                # For open spaces, use Shapely distance from room center to polygon
                distance = Point(room_center_x, room_center_y).distance(segment.polygon)
                if distance < best_distance:
                    best_distance = distance
                    best_segment = segment
            elif segment.is_horizontal:
                seg_minx = min(segment.start[0], segment.end[0])
                seg_maxx = max(segment.start[0], segment.end[0])
                if seg_minx <= room_center_x <= seg_maxx:
                    distance = abs(room_center_y - segment.start[1])
                    if distance < best_distance:
                        best_distance = distance
                        best_segment = segment
            else:
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
