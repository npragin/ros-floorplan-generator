"""Hallway segment collision detection and path planning."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from shapely.geometry import Polygon, box

from floorplan_generator.geometry import (
    Direction,
    get_perpendicular_offset_directions,
    move_in_direction,
    opposite_direction,
    turn_direction,
)


class CollisionError(Exception):
    """Raised when no collision-free hallway path can be found."""


@dataclass
class PlannedSegment:
    """Lightweight path segment for collision checking before full construction."""

    index: int
    start: tuple[float, float]
    end: tuple[float, float]
    direction: Direction
    length: float
    is_open_space: bool = False
    side_length: float = 0.0
    expand_direction: Direction | None = None

    def bounding_polygon(self, hallway_width: float) -> Polygon:
        """
        Build the Shapely box for this segment.

        For regular segments, produces a symmetric hallway-width rectangle.
        For open space segments with a known expand_direction, produces the
        asymmetric polygon matching ``create_open_space_segment()``.
        """
        half_width = hallway_width / 2
        x1, y1 = self.start
        x2, y2 = self.end

        if not self.is_open_space or self.expand_direction is None:
            if self.direction in ("east", "west"):
                min_x, max_x = min(x1, x2), max(x1, x2)
                return box(min_x, y1 - half_width, max_x, y1 + half_width)
            else:
                min_y, max_y = min(y1, y2), max(y1, y2)
                return box(x1 - half_width, min_y, x1 + half_width, max_y)

        large_offset = self.length - half_width

        if self.direction in ("east", "west"):
            min_x, max_x = min(x1, x2), max(x1, x2)
            center_y = y1
            if self.expand_direction == "north":
                return box(min_x, center_y - half_width, max_x, center_y + large_offset)
            else:
                return box(min_x, center_y - large_offset, max_x, center_y + half_width)
        else:
            min_y, max_y = min(y1, y2), max(y1, y2)
            center_x = x1
            if self.expand_direction == "east":
                return box(center_x - half_width, min_y, center_x + large_offset, max_y)
            else:
                return box(center_x - large_offset, min_y, center_x + half_width, max_y)


class CollisionChecker:
    """Checks whether a new hallway segment overlaps existing segments."""

    def __init__(
        self,
        hallway_width: float,
        allowed_connections: set[tuple[int, int]] | None = None,
        tolerance: float = 0.01,
    ) -> None:
        """
        Initialize the collision checker.

        Args:
            hallway_width: Width of the hallway corridor.
            allowed_connections: Set of (i, j) segment index pairs allowed to overlap.
            tolerance: Minimum intersection area to count as a collision.

        """
        self.hallway_width = hallway_width
        self.allowed_connections = allowed_connections or set()
        self.tolerance = tolerance

    def check_segment(
        self,
        new_segment: PlannedSegment,
        existing_segments: list[PlannedSegment],
    ) -> bool:
        """Return True if the new segment collides with any existing segment."""
        new_poly = new_segment.bounding_polygon(self.hallway_width)

        for existing in existing_segments:
            # Skip adjacent segments (they share endpoints at corners)
            if abs(new_segment.index - existing.index) == 1:
                continue

            # Skip allowed connections
            pair = (min(new_segment.index, existing.index), max(new_segment.index, existing.index))
            if pair in self.allowed_connections:
                continue

            existing_poly = existing.bounding_polygon(self.hallway_width)
            intersection = new_poly.intersection(existing_poly)
            if intersection.area > self.tolerance:
                return True

        return False


def _compute_expand_direction(
    seg_direction: Direction,
    adjacent_direction: Direction,
) -> Direction:
    """
    Compute expand direction: away from an adjacent segment's body.

    The adjacent segment's body extends in ``adjacent_direction`` from the
    junction. We expand to the perpendicular side of ``seg_direction`` that
    is farthest from that body.
    """
    left_dir, right_dir = get_perpendicular_offset_directions(seg_direction)
    body_side = opposite_direction(adjacent_direction)
    if body_side == left_dir:
        return right_dir
    return left_dir


def _direction_vector(direction: Direction) -> tuple[float, float]:
    """Return the unit vector for a cardinal direction."""
    vectors: dict[str, tuple[float, float]] = {
        "east": (1.0, 0.0),
        "west": (-1.0, 0.0),
        "north": (0.0, 1.0),
        "south": (0.0, -1.0),
    }
    return vectors[direction]


def _compute_attachment_offset(
    prev_segment: "PlannedSegment",
    next_direction: Direction,
    hallway_width: float,
) -> tuple[float, float]:
    """
    Compute the offset from prev_segment.end to the next segment's start.

    Adjacent segments don't overlap. The next segment starts past the
    previous segment's end: half the hallway width forward (clearing the
    prev segment's body) plus half the hallway width sideways (centering
    the next segment on its own centerline).

    For open spaces the sideways offset uses the full asymmetric extent
    when the next direction matches the expand direction.

    Returns an (dx, dy) offset to add to the prev segment's end position.
    """
    half_width = hallway_width / 2

    # Step forward past the end of the previous segment
    forward = _direction_vector(prev_segment.direction)

    # Step sideways to center the next segment's centerline
    if prev_segment.is_open_space and prev_segment.expand_direction is not None:
        sideways_amount = (
            prev_segment.length - half_width if next_direction == prev_segment.expand_direction else half_width
        )
    else:
        sideways_amount = half_width
    sideways = _direction_vector(next_direction)

    return (
        half_width * forward[0] + sideways_amount * sideways[0],
        half_width * forward[1] + sideways_amount * sideways[1],
    )


def plan_path(
    start_pos: tuple[float, float],
    start_direction: Direction,
    num_segments: int,
    segment_length_fn: Callable[[int, str | None, Direction], float],
    turn_chooser: Callable[[int], str],
    hallway_width: float,
    allowed_connections: set[tuple[int, int]] | None = None,
    open_space_indices: set[int] | None = None,
) -> tuple[list[PlannedSegment], list[str]]:
    """
    Plan a collision-free hallway path using backtracking.

    Args:
        start_pos: Starting position of the first segment.
        start_direction: Initial direction of travel.
        num_segments: Total number of segments to plan.
        segment_length_fn: Callable(seg_index, turn_at_start_dir, direction) -> length.
        turn_chooser: Callable(turn_index) -> "left" or "right" (initial choice).
        hallway_width: Width of the hallway corridor.
        allowed_connections: Set of (i, j) segment index pairs allowed to overlap.
        open_space_indices: Segment indices that will become open spaces.

    Returns:
        A tuple of (list of PlannedSegments, list of turn directions used).

    Raises:
        CollisionError: If no valid path exists.

    """
    checker = CollisionChecker(hallway_width, allowed_connections)
    os_indices = open_space_indices or set()

    # Each level of the stack: (seg_index, tried_turns)
    # tried_turns tracks which turn choices have been attempted at this level
    segments: list[PlannedSegment] = []
    turns_used: list[str] = []

    # For each turn point, track which choices we've tried
    # turn_choices[i] = list of turns tried at turn index i
    turn_choices: list[list[str]] = []

    seg_idx = 0
    while seg_idx < num_segments:
        if seg_idx == 0:
            # First segment: no turn needed
            direction = start_direction
            pos = start_pos
            turn_at_start_dir = None
        else:
            # Need to pick a turn
            turn_idx = seg_idx - 1

            # Initialize choices for this turn if needed
            while len(turn_choices) <= turn_idx:
                turn_choices.append([])

            # Get available turn options
            if not turn_choices[turn_idx]:
                # First attempt at this turn: use turn_chooser's suggestion first
                initial = turn_chooser(turn_idx)
                opposite = "right" if initial == "left" else "left"
                available = [initial, opposite]
            else:
                # We're backtracking - see what's left
                initial = turn_chooser(turn_idx)
                opposite = "right" if initial == "left" else "left"
                all_options = [initial, opposite]
                available = [t for t in all_options if t not in turn_choices[turn_idx]]

            if not available:
                # Exhausted both options at this turn, backtrack further
                turn_choices[turn_idx] = []  # Reset for future visits
                if seg_idx <= 1:
                    raise CollisionError("No collision-free path found: exhausted all turn combinations.")
                # Remove the previous segment and turn
                segments.pop()
                turns_used.pop()
                seg_idx -= 1
                continue

            chosen_turn = available[0]
            turn_choices[turn_idx].append(chosen_turn)

            prev_segment = segments[-1]
            turn_lr: Literal["left", "right"] = "left" if chosen_turn == "left" else "right"
            direction = turn_direction(prev_segment.direction, turn_lr)
            turn_at_start_dir = chosen_turn

            # Offset so the next segment starts past the previous one (no overlap)
            dx, dy = _compute_attachment_offset(prev_segment, direction, hallway_width)
            pos = (prev_segment.end[0] + dx, prev_segment.end[1] + dy)

        # Calculate length and build planned segment
        length = segment_length_fn(seg_idx, turn_at_start_dir if seg_idx > 0 else None, direction)
        end = move_in_direction(pos, direction, length)

        is_open = seg_idx in os_indices
        planned = PlannedSegment(
            index=seg_idx,
            start=pos,
            end=end,
            direction=direction,
            length=length,
            is_open_space=is_open,
            side_length=length if is_open else 0.0,
        )

        # Compute expand direction for open space segments
        if is_open:
            if seg_idx > 0:
                planned.expand_direction = _compute_expand_direction(direction, segments[-1].direction)
            else:
                # First segment: nothing to collide with, default to left of travel
                planned.expand_direction = get_perpendicular_offset_directions(direction)[0]

        # Check collision against existing segments
        if checker.check_segment(planned, segments):
            # Collision detected - if this is the first segment we can't backtrack
            if seg_idx == 0:
                raise CollisionError("First segment collides (impossible in normal usage).")
            # Try the next available turn (loop back to top)
            continue

        # No collision, commit this segment
        segments.append(planned)
        if seg_idx > 0:
            turns_used.append(turn_at_start_dir)  # type: ignore[arg-type]
        seg_idx += 1

    return segments, turns_used
