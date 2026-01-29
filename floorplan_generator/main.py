"""Command-line interface for floorplan generation."""

import random
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

from floorplan_generator.generator import FloorplanGenerator
from floorplan_generator.params import FloorplanParams
from floorplan_generator.renderer import FloorplanRenderer
from floorplan_generator.validator import LayoutValidator

app = typer.Typer(
    add_completion=False,
)

# Default values for floorplan parameters
DEFAULTS: dict[str, Any] = {
    "doorway_width": 1.0,
    "hallway_width": 2.0,
    "room_wall_length": 4.0,
    "wall_thickness": 0.2,
    "room_spacing": 0.0,
    "doorway_length": 0.0,
    "hallway_end_padding": 0.0,
    "num_turns": 0,
    "num_open_spaces": 0,
    "turn_direction": "alternating",
    "seed": None,
    "max_retries": 10,
    "output": "output/floorplan.png",
    "resolution": 0.05,
    "debug": False,
    "skip_validation": False,
}


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Dictionary of configuration values.

    Raises:
        typer.BadParameter: If the config file cannot be read or parsed.

    """
    try:
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise typer.BadParameter(f"Config file not found: {config_path}") from None
    except tomllib.TOMLDecodeError as e:
        raise typer.BadParameter(f"Invalid TOML in config file: {e}") from None


def resolve_param(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Resolve a parameter value with priority: CLI > config > default.

    Args:
        cli_value: Value passed via CLI (None if not provided).
        config: Configuration dictionary from TOML file.
        key: Parameter key name.
        default: Default value if not in CLI or config.

    Returns:
        The resolved parameter value.

    """
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


@app.command()
def generate(
    # Config file option
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to TOML configuration file. CLI arguments override config file values.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    # Floorplan parameters
    num_rooms: Annotated[
        int | None,
        typer.Option(
            "--num-rooms",
            "-n",
            help="Number of rooms to generate.",
        ),
    ] = None,
    doorway_width: Annotated[
        float | None,
        typer.Option(
            "--doorway-width",
            "-d",
            help="Width of door openings in meters.",
            show_default=str(DEFAULTS["doorway_width"]),
        ),
    ] = None,
    hallway_width: Annotated[
        float | None,
        typer.Option(
            "--hallway-width",
            "-w",
            help="Width of hallway corridors in meters.",
            show_default=str(DEFAULTS["hallway_width"]),
        ),
    ] = None,
    room_wall_length: Annotated[
        float | None,
        typer.Option(
            "--room-wall-length",
            "-r",
            help="Side length of square rooms in meters.",
            show_default=str(DEFAULTS["room_wall_length"]),
        ),
    ] = None,
    wall_thickness: Annotated[
        float | None,
        typer.Option(
            "--wall-thickness",
            "-t",
            help="Thickness of all walls in meters.",
            show_default=str(DEFAULTS["wall_thickness"]),
        ),
    ] = None,
    room_spacing: Annotated[
        float | None,
        typer.Option(
            "--room-spacing",
            "-s",
            help="Gap between adjacent rooms along the hallway in meters.",
            show_default=str(DEFAULTS["room_spacing"]),
        ),
    ] = None,
    doorway_length: Annotated[
        float | None,
        typer.Option(
            "--doorway-length",
            "-l",
            help="Length/depth of door passage in meters. If 0, just cuts through walls.",
            show_default=str(DEFAULTS["doorway_length"]),
        ),
    ] = None,
    hallway_end_padding: Annotated[
        float | None,
        typer.Option(
            "--hallway-end-padding",
            "-p",
            help="Padding added to each end of the hallway in meters. When 0, hallway ends align with room edges.",
            show_default=str(DEFAULTS["hallway_end_padding"]),
        ),
    ] = None,
    # Turn parameters
    num_turns: Annotated[
        int | None,
        typer.Option(
            "--num-turns",
            help="Number of 90-degree turns in the hallway. 0 for straight hallway.",
            show_default=str(DEFAULTS["num_turns"]),
        ),
    ] = None,
    num_open_spaces: Annotated[
        int | None,
        typer.Option(
            "--num-open-spaces",
            help="Number of hallway segments to convert to open spaces. Must be <= num_turns + 1.",
            show_default=str(DEFAULTS["num_open_spaces"]),
        ),
    ] = None,
    turn_direction: Annotated[
        str | None,
        typer.Option(
            "--turn-direction",
            help="Direction pattern for turns: 'alternating', 'random', 'clockwise', or 'counterclockwise'.",
            show_default=str(DEFAULTS["turn_direction"]),
        ),
    ] = None,
    # Seed and retry options
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for reproducible generation. If not set, system entropy is used.",
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option(
            "--max-retries",
            help="Maximum retry attempts when collision-free path cannot be found.",
            show_default=str(DEFAULTS["max_retries"]),
        ),
    ] = None,
    # Output options
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PNG file path.",
            show_default=str(DEFAULTS["output"]),
        ),
    ] = None,
    resolution: Annotated[
        float | None,
        typer.Option(
            help="Meters per pixel (e.g., 0.05 = 20 pixels per meter).",
            show_default=str(DEFAULTS["resolution"]),
        ),
    ] = None,
    debug: Annotated[
        bool | None,
        typer.Option(
            "--debug/--no-debug",
            help="Generate a debug visualization.",
            show_default="no-debug",
        ),
    ] = None,
    debug_steps: Annotated[
        str | None,
        typer.Option(
            "--debug-steps",
            help="Directory to save debug images for each generation step.",
        ),
    ] = None,
    skip_validation: Annotated[
        bool | None,
        typer.Option(
            "--skip-validation/--no-skip-validation",
            help="Skip validation checks.",
            show_default="no-skip-validation",
        ),
    ] = None,
) -> None:
    """
    Generate office-style floorplans for ROS2 Stage simulator.

    Configuration priority: CLI arguments > config file > defaults
    """
    # Load config file if provided
    cfg: dict[str, Any] = {}
    if config is not None:
        cfg = load_config(config)
        typer.echo(f"Loaded configuration from {config}")

    # Resolve all parameters with priority: CLI > config > defaults
    resolved_num_rooms = resolve_param(num_rooms, cfg, "num_rooms")
    resolved_doorway_width = resolve_param(doorway_width, cfg, "doorway_width", DEFAULTS["doorway_width"])
    resolved_hallway_width = resolve_param(hallway_width, cfg, "hallway_width", DEFAULTS["hallway_width"])
    resolved_room_wall_length = resolve_param(room_wall_length, cfg, "room_wall_length", DEFAULTS["room_wall_length"])
    resolved_wall_thickness = resolve_param(wall_thickness, cfg, "wall_thickness", DEFAULTS["wall_thickness"])
    resolved_room_spacing = resolve_param(room_spacing, cfg, "room_spacing", DEFAULTS["room_spacing"])
    resolved_doorway_length = resolve_param(doorway_length, cfg, "doorway_length", DEFAULTS["doorway_length"])
    resolved_hallway_end_padding = resolve_param(
        hallway_end_padding, cfg, "hallway_end_padding", DEFAULTS["hallway_end_padding"]
    )
    resolved_num_turns = resolve_param(num_turns, cfg, "num_turns", DEFAULTS["num_turns"])
    resolved_num_open_spaces = resolve_param(num_open_spaces, cfg, "num_open_spaces", DEFAULTS["num_open_spaces"])
    resolved_turn_direction = resolve_param(turn_direction, cfg, "turn_direction", DEFAULTS["turn_direction"])
    resolved_seed = resolve_param(seed, cfg, "seed", DEFAULTS["seed"])
    resolved_max_retries = resolve_param(max_retries, cfg, "max_retries", DEFAULTS["max_retries"])
    resolved_output = resolve_param(output, cfg, "output", DEFAULTS["output"])
    resolved_resolution = resolve_param(resolution, cfg, "resolution", DEFAULTS["resolution"])
    resolved_debug = resolve_param(debug, cfg, "debug", DEFAULTS["debug"])
    resolved_debug_steps = resolve_param(debug_steps, cfg, "debug_steps")
    resolved_skip_validation = resolve_param(skip_validation, cfg, "skip_validation", DEFAULTS["skip_validation"])

    # Validate required parameters
    if resolved_num_rooms is None:
        raise typer.BadParameter("num_rooms is required. Provide it via --num-rooms/-n or in a config file.")

    # Handle seed: auto-generate if not provided, then seed the RNG once here
    seed_was_provided = resolved_seed is not None
    if resolved_seed is None:
        resolved_seed = random.randrange(2**32)
    random.seed(resolved_seed)

    # If user explicitly chose a seed, don't retry (results should be deterministic)
    if seed_was_provided:
        resolved_max_retries = 0

    # Create parameters
    params = FloorplanParams(
        doorway_width=resolved_doorway_width,
        num_rooms=resolved_num_rooms,
        hallway_width=resolved_hallway_width,
        room_wall_length=resolved_room_wall_length,
        wall_thickness=resolved_wall_thickness,
        room_spacing=resolved_room_spacing,
        doorway_length=resolved_doorway_length,
        hallway_end_padding=resolved_hallway_end_padding,
        num_turns=resolved_num_turns,
        num_open_spaces=resolved_num_open_spaces,
        turn_direction=resolved_turn_direction,
        seed=resolved_seed,
    )

    typer.echo(f"Generating floorplan with {params.num_rooms} rooms...")
    typer.echo(f"  Room size: {params.room_wall_length}m x {params.room_wall_length}m")
    typer.echo(f"  Hallway width: {params.hallway_width}m")
    typer.echo(f"  Door width: {params.doorway_width}m")
    typer.echo(f"  Door length: {params.effective_doorway_length}m")
    typer.echo(f"  Wall thickness: {params.wall_thickness}m")
    typer.echo(f"  Room spacing: {params.room_spacing}m")
    typer.echo(f"  Hallway end padding: {params.hallway_end_padding}m")
    typer.echo(f"  Number of turns: {params.num_turns}")
    typer.echo(f"  Open spaces: {params.num_open_spaces}")
    typer.echo(f"  Turn direction: {params.turn_direction}")
    if seed_was_provided:
        typer.echo(f"  Seed: {resolved_seed}")
    else:
        typer.echo(f"  Seed: {resolved_seed} (auto-generated, use --seed {resolved_seed} to reproduce)")
    typer.echo(f"  Max retries: {resolved_max_retries}")

    # Generate floorplan
    generator = FloorplanGenerator(params, max_retries=resolved_max_retries)
    floorplan = generator.generate(debug_dir=resolved_debug_steps)

    # Validate (defer output until end)
    validation_errors: list[str] = []
    if not resolved_skip_validation:
        validator = LayoutValidator()
        _, validation_errors = validator.validate_all(floorplan)

    # Render
    output_path = Path(resolved_output)
    typer.echo(f"\nRendering to {output_path}...")

    renderer = FloorplanRenderer(resolution=resolved_resolution)
    renderer.render_to_png(floorplan, output_path)
    typer.echo(f"  Saved occupancy grid to {output_path}")

    if resolved_debug:
        debug_path = output_path.with_stem(output_path.stem + "_debug")
        renderer.render_debug(floorplan, debug_path)
        typer.echo(f"  Saved debug visualization to {debug_path}")

    # Print some stats
    bounds = floorplan.get_bounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    typer.echo(f"\nFloorplan dimensions: {width:.1f}m x {height:.1f}m")

    # Print validation results at the end (colored for visibility)
    if not resolved_skip_validation:
        if not validation_errors:
            typer.secho("\nValidation: All checks passed!", fg=typer.colors.GREEN)
        else:
            typer.secho("\n" + "=" * 50, fg=typer.colors.RED, bold=True)
            typer.secho("VALIDATION ERRORS", fg=typer.colors.RED, bold=True)
            typer.secho("=" * 50, fg=typer.colors.RED, bold=True)
            for error in validation_errors:
                typer.secho(f"  • {error}", fg=typer.colors.RED)
            typer.secho("=" * 50, fg=typer.colors.RED, bold=True)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
