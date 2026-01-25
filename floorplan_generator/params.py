"""Parameter definitions for floorplan generation."""

from dataclasses import dataclass


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

    """

    doorway_width: float
    num_rooms: int
    hallway_width: float
    room_wall_length: float
    wall_thickness: float
    room_spacing: float = 0.0
    doorway_length: float = 0.0

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

        # Check doorway fits in hallway
        if self.doorway_width >= self.hallway_width:
            raise ValueError(
                f"doorway_width ({self.doorway_width}) must be less than hallway_width ({self.hallway_width})"
            )

        # Check doorway fits in room wall
        if self.doorway_width >= self.room_wall_length:
            raise ValueError(
                f"doorway_width ({self.doorway_width}) must be less than room_wall_length ({self.room_wall_length})"
            )

        # Check wall thickness is reasonable
        if self.wall_thickness >= self.room_wall_length / 2:
            raise ValueError(
                f"wall_thickness ({self.wall_thickness}) must be less than "
                f"half the room_wall_length ({self.room_wall_length / 2})"
            )

        # Check doorway_length is non-negative
        if self.doorway_length < 0:
            raise ValueError(f"doorway_length must be non-negative, got {self.doorway_length}")

    @property
    def effective_doorway_length(self) -> float:
        """
        Get the effective doorway length, accounting for minimum wall cut-through.

        Returns:
            The doorway length to use, at minimum enough to cut through walls.

        """
        return max(self.doorway_length, self.wall_thickness)
