"""Command-line interface for floorplan generation."""

import argparse
from pathlib import Path

from floorplan_generator.generator import FloorplanGenerator
from floorplan_generator.params import FloorplanParams
from floorplan_generator.renderer import FloorplanRenderer
from floorplan_generator.validator import LayoutValidator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(
        description="Generate office-style floorplans for ROS2 Stage simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required parameters
    parser.add_argument(
        "-n",
        "--num-rooms",
        type=int,
        required=True,
        help="Number of rooms to generate",
    )

    # Optional parameters with defaults
    parser.add_argument(
        "-d",
        "--doorway-width",
        type=float,
        default=1.0,
        help="Width of door openings (meters)",
    )
    parser.add_argument(
        "-w",
        "--hallway-width",
        type=float,
        default=2.0,
        help="Width of hallway corridors (meters)",
    )
    parser.add_argument(
        "-r",
        "--room-wall-length",
        type=float,
        default=4.0,
        help="Side length of square rooms (meters)",
    )
    parser.add_argument(
        "-t",
        "--wall-thickness",
        type=float,
        default=0.2,
        help="Thickness of all walls (meters)",
    )
    parser.add_argument(
        "-s",
        "--room-spacing",
        type=float,
        default=0.0,
        help="Gap between adjacent rooms along the hallway (meters)",
    )
    parser.add_argument(
        "-l",
        "--doorway-length",
        type=float,
        default=0.0,
        help="Length/depth of door passage (meters). If 0, just cuts through walls.",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/floorplan.png",
        help="Output PNG file path",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Meters per pixel (e.g., 0.05 = 20 pixels per meter)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also generate a debug visualization",
    )
    parser.add_argument(
        "--debug-steps",
        type=str,
        default=None,
        help="Directory to save debug images for each generation step",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks",
    )

    return parser.parse_args()


def main() -> None:
    """Run the floorplan generator."""
    args = parse_args()

    # Create parameters
    params = FloorplanParams(
        doorway_width=args.doorway_width,
        num_rooms=args.num_rooms,
        hallway_width=args.hallway_width,
        room_wall_length=args.room_wall_length,
        wall_thickness=args.wall_thickness,
        room_spacing=args.room_spacing,
        doorway_length=args.doorway_length,
    )

    print(f"Generating floorplan with {params.num_rooms} rooms...")
    print(f"  Room size: {params.room_wall_length}m x {params.room_wall_length}m")
    print(f"  Hallway width: {params.hallway_width}m")
    print(f"  Door width: {params.doorway_width}m")
    print(f"  Door length: {params.effective_doorway_length}m")
    print(f"  Wall thickness: {params.wall_thickness}m")
    print(f"  Room spacing: {params.room_spacing}m")

    # Generate floorplan
    generator = FloorplanGenerator(params)
    floorplan = generator.generate(debug_dir=args.debug_steps)

    # Validate
    if not args.skip_validation:
        print("\nValidating floorplan...")
        validator = LayoutValidator()
        is_valid, errors = validator.validate_all(floorplan)
        if is_valid:
            print("  All validations passed!")
        else:
            print("  Validation errors:")
            for error in errors:
                print(f"    - {error}")
            print("\nContinuing with rendering despite validation errors...")

    # Render
    output_path = Path(args.output)
    print(f"\nRendering to {output_path}...")

    renderer = FloorplanRenderer(resolution=args.resolution)
    renderer.render_to_png(floorplan, output_path)
    print(f"  Saved occupancy grid to {output_path}")

    if args.debug:
        debug_path = output_path.with_stem(output_path.stem + "_debug")
        renderer.render_debug(floorplan, debug_path)
        print(f"  Saved debug visualization to {debug_path}")

    # Print some stats
    bounds = floorplan.get_bounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    print(f"\nFloorplan dimensions: {width:.1f}m x {height:.1f}m")


if __name__ == "__main__":
    main()
