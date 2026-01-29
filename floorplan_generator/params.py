"""Parameter definitions for floorplan generation."""

from dataclasses import dataclass
from typing import Literal

TurnDirection = Literal["alternating", "random", "clockwise", "counterclockwise"]


@dataclass
class FloorplanParams:
    """
    Parameters for generating a floorplan.

    Attributes:
        doorway_width: Width of door openings in meters.
        num_rooms: Total number of rooms to generate.
        hallway_width: Width of hallway corridors in meters.
        room_wall_length: Side length of square rooms in meters.
        wall_thickness: Thickness of all walls in meters.
        room_spacing: Gap between adjacent rooms along the hallway in meters.
        doorway_length: Length/depth of door passage in meters. If <= wall_thickness,
            the doorway just cuts through the wall. If larger, creates an extended passage.
        hallway_end_padding: Padding added to each end of the hallway in meters.
            When set to 0, the hallway ends align with the edges of the first and last rooms.
        num_turns: Number of 90-degree turns in the hallway. 0 for straight hallway.
        turn_direction: Direction pattern for turns. Options:
            - "alternating": alternates between left and right turns
            - "random": random turn direction at each turn
            - "clockwise": always turn right (clockwise)
            - "counterclockwise": always turn left (counterclockwise)
        num_open_spaces: Number of hallway segments to convert to open spaces (squares).
            Must be <= num_turns + 1.
        seed: Random seed for reproducible generation. When None, system entropy is used.

    """

    doorway_width: float
    num_rooms: int
    hallway_width: float
    room_wall_length: float
    wall_thickness: float
    room_spacing: float = 0.0
    doorway_length: float = 0.0
    hallway_end_padding: float = 0.0
    num_turns: int = 0
    turn_direction: TurnDirection = "alternating"
    num_open_spaces: int = 0
    seed: int | None = None

    def validate(self) -> None:
        """
        Validate parameter constraints.

        Raises:
            ValueError: If any parameter constraint is violated.

        """
        # Check positive values
        if self.doorway_width <= 0:
            raise ValueError(f"doorway_width must be positive, got {self.doorway_width}")
        if self.num_rooms <= 0:
            raise ValueError(f"num_rooms must be positive, got {self.num_rooms}")
        if self.hallway_width <= 0:
            raise ValueError(f"hallway_width must be positive, got {self.hallway_width}")
        if self.room_wall_length <= 0:
            raise ValueError(f"room_wall_length must be positive, got {self.room_wall_length}")
        if self.wall_thickness <= 0:
            raise ValueError(f"wall_thickness must be positive, got {self.wall_thickness}")
        if self.room_spacing < 0:
            raise ValueError(f"room_spacing must be non-negative, got {self.room_spacing}")

        # Check doorway fits in room wall
        if self.doorway_width > self.room_wall_length:
            raise ValueError(
                f"doorway_width ({self.doorway_width}) must be less than or equal to room_wall_length \
                    ({self.room_wall_length})"
            )

        # Check doorway_length is non-negative
        if self.doorway_length < 0:
            raise ValueError(f"doorway_length must be non-negative, got {self.doorway_length}")

        # Check hallway_end_padding is non-negative
        if self.hallway_end_padding < 0:
            raise ValueError(f"hallway_end_padding must be non-negative, got {self.hallway_end_padding}")

        # Check num_turns is non-negative
        if self.num_turns < 0:
            raise ValueError(f"num_turns must be non-negative, got {self.num_turns}")

        # Check num_open_spaces
        if self.num_open_spaces < 0:
            raise ValueError(f"num_open_spaces must be non-negative, got {self.num_open_spaces}")
        if self.num_open_spaces > self.num_turns + 1:
            raise ValueError(
                f"num_open_spaces ({self.num_open_spaces}) must be <= num_turns + 1 ({self.num_turns + 1})"
            )

        # Check turn_direction is valid
        valid_turn_directions = ("alternating", "random", "clockwise", "counterclockwise")
        if self.turn_direction not in valid_turn_directions:
            raise ValueError(f"turn_direction must be one of {valid_turn_directions}, got '{self.turn_direction}'")

    @property
    def min_segment_length(self) -> float:
        """
        Get the minimum segment length based on room dimensions.

        A segment must be long enough to fit at least one room on each side,
        accounting for the doorway corridor gap and wall thickness. For middle
        segments with turns at both ends, the turn buffer is used instead of
        hallway_end_padding.

        Returns:
            The minimum segment length in meters.

        """
        return self.room_wall_length

    @property
    def effective_doorway_length(self) -> float:
        """
        Get the effective doorway length, accounting for minimum wall cut-through.

        Returns:
            The doorway length to use, at minimum enough to cut through walls.

        """
        return max(self.doorway_length, self.wall_thickness)
