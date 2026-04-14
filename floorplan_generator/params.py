"""Parameter definitions for floorplan generation."""

from dataclasses import dataclass
from typing import Literal

TurnDirection = Literal["alternating", "random", "clockwise", "counterclockwise"]
ObstaclePlacement = Literal["rooms", "hallways", "both"]


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
        clustered_robot_spawns: If True (default), robots are placed using greedy circle packing
            starting from a random valid position, clustering them together. If False, all robot
            positions are chosen independently at random.

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
    obstacles_enabled: bool = False
    num_obstacles: int = 0
    obstacle_length: float = 1.0
    obstacle_clearance: float = 0.5
    obstacle_spacing: float = 0.5
    obstacle_placement: ObstaclePlacement = "both"
    robot_radius: float | None = None
    num_robots: int | None = None
    robot_min_clearance: float | None = None
    spawn_export_filename: str | None = None
    num_extra_points: int | None = None
    extra_point_radius: float | None = None
    extra_point_min_clearance: float | None = None
    clustered_robot_spawns: bool = True

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

        # Check obstacle parameters
        if self.obstacles_enabled:
            if self.num_obstacles < 0:
                raise ValueError(f"num_obstacles must be non-negative, got {self.num_obstacles}")
            if self.obstacle_length <= 0:
                raise ValueError(f"obstacle_length must be positive, got {self.obstacle_length}")
            if self.obstacle_clearance < 0:
                raise ValueError(f"obstacle_clearance must be non-negative, got {self.obstacle_clearance}")
            if self.obstacle_spacing < 0:
                raise ValueError(f"obstacle_spacing must be non-negative, got {self.obstacle_spacing}")
            valid_placements = ("rooms", "hallways", "both")
            if self.obstacle_placement not in valid_placements:
                raise ValueError(
                    f"obstacle_placement must be one of {valid_placements}, got '{self.obstacle_placement}'"
                )

        # Check robot spawn parameters: all three required params must be set or all None
        spawn_params = [self.robot_radius, self.num_robots, self.robot_min_clearance]
        spawn_set = [p is not None for p in spawn_params]
        if any(spawn_set) and not all(spawn_set):
            raise ValueError("robot_radius, num_robots, and robot_min_clearance must all be provided or all be None")
        if all(spawn_set):
            if self.robot_radius <= 0:  # type: ignore[operator]
                raise ValueError(f"robot_radius must be positive, got {self.robot_radius}")
            if self.num_robots <= 0:  # type: ignore[operator]
                raise ValueError(f"num_robots must be positive, got {self.num_robots}")
            if self.robot_min_clearance < 0:  # type: ignore[operator]
                raise ValueError(f"robot_min_clearance must be non-negative, got {self.robot_min_clearance}")

        # Check extra point parameters: all three must be set or all None
        extra_params = [self.num_extra_points, self.extra_point_radius, self.extra_point_min_clearance]
        extra_set = [p is not None for p in extra_params]
        if any(extra_set) and not all(extra_set):
            raise ValueError(
                "num_extra_points, extra_point_radius, and extra_point_min_clearance must all be provided or all be None"
            )
        if all(extra_set):
            if self.num_extra_points <= 0:  # type: ignore[operator]
                raise ValueError(f"num_extra_points must be positive, got {self.num_extra_points}")
            if self.extra_point_radius <= 0:  # type: ignore[operator]
                raise ValueError(f"extra_point_radius must be positive, got {self.extra_point_radius}")
            if self.extra_point_min_clearance < 0:  # type: ignore[operator]
                raise ValueError(f"extra_point_min_clearance must be non-negative, got {self.extra_point_min_clearance}")

    def min_segment_length(self, is_end_segment: bool) -> float:
        """
        Get the minimum segment length based on room dimensions.

        A segment must be long enough to fit at least one room on each side,
        accounting for the doorway corridor gap and wall thickness. For middle
        segments with turns at both ends, the turn buffer is used instead of
        hallway_end_padding.

        Returns:
            The minimum segment length in meters.

        """
        if is_end_segment:
            return self.room_wall_length + self.effective_doorway_length
        else:
            return self.room_wall_length + self.effective_doorway_length * 2

    @property
    def effective_doorway_length(self) -> float:
        """
        Get the effective doorway length, accounting for minimum wall cut-through.

        Returns:
            The doorway length to use, at minimum enough to cut through walls.

        """
        return max(self.doorway_length, self.wall_thickness)
