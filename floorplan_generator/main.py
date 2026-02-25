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
    "obstacles_enabled": False,
    "num_obstacles": 0,
    "obstacle_length": 1.0,
    "obstacle_clearance": 0.5,
    "obstacle_spacing": 0.5,
    "obstacle_placement": "both",
    "robot_radius": None,
    "num_robots": None,
    "robot_min_clearance": None,
    "spawn_export_filename": None,
    "num_extra_points": None,
    "extra_point_radius": None,
    "extra_point_min_clearance": None,
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
            help="Number of hallway segments to convert to open spaces. Must be <= (num_turns + 1) / 2.",
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
    # Obstacle parameters
    obstacles: Annotated[
        bool | None,
        typer.Option(
            "--obstacles/--no-obstacles",
            help="Enable obstacle generation.",
            show_default="no-obstacles",
        ),
    ] = None,
    num_obstacles: Annotated[
        int | None,
        typer.Option(
            "--num-obstacles",
            help="Number of obstacles to place in the floorplan.",
            show_default=str(DEFAULTS["num_obstacles"]),
        ),
    ] = None,
    obstacle_length: Annotated[
        float | None,
        typer.Option(
            "--obstacle-length",
            help="Side length of square obstacles in meters.",
            show_default=str(DEFAULTS["obstacle_length"]),
        ),
    ] = None,
    obstacle_clearance: Annotated[
        float | None,
        typer.Option(
            "--obstacle-clearance",
            help="Minimum clearance around obstacles in meters.",
            show_default=str(DEFAULTS["obstacle_clearance"]),
        ),
    ] = None,
    obstacle_spacing: Annotated[
        float | None,
        typer.Option(
            "--obstacle-spacing",
            help="Minimum distance between obstacles in meters.",
            show_default=str(DEFAULTS["obstacle_spacing"]),
        ),
    ] = None,
    obstacle_placement: Annotated[
        str | None,
        typer.Option(
            "--obstacle-placement",
            help="Where to place obstacles: 'rooms', 'hallways', or 'both'.",
            show_default=str(DEFAULTS["obstacle_placement"]),
        ),
    ] = None,
    # Robot spawn parameters
    robot_radius: Annotated[
        float | None,
        typer.Option(
            "--robot-radius",
            help="Robot radius in meters. Required with --num-robots and --robot-min-clearance.",
        ),
    ] = None,
    num_robots: Annotated[
        int | None,
        typer.Option(
            "--num-robots",
            help="Number of robots to generate spawn positions for.",
        ),
    ] = None,
    robot_min_clearance: Annotated[
        float | None,
        typer.Option(
            "--robot-min-clearance",
            help="Minimum clearance from walls and between robot edges in meters.",
        ),
    ] = None,
    spawn_export_filename: Annotated[
        str | None,
        typer.Option(
            "--spawn-export-filename",
            help="Filename for world config YAML output. Defaults to 'world_config.yaml'.",
        ),
    ] = None,
    # Extra point parameters
    num_extra_points: Annotated[
        int | None,
        typer.Option(
            "--num-extra-points",
            help="Number of extra points to place in free space.",
        ),
    ] = None,
    extra_point_radius: Annotated[
        float | None,
        typer.Option(
            "--extra-point-radius",
            help="Radius of each extra point in meters.",
        ),
    ] = None,
    extra_point_min_clearance: Annotated[
        float | None,
        typer.Option(
            "--extra-point-min-clearance",
            help="Minimum clearance from walls and between extra point edges in meters.",
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
    resolved_obstacles_enabled = resolve_param(obstacles, cfg, "obstacles_enabled", DEFAULTS["obstacles_enabled"])
    resolved_num_obstacles = resolve_param(num_obstacles, cfg, "num_obstacles", DEFAULTS["num_obstacles"])
    resolved_obstacle_length = resolve_param(obstacle_length, cfg, "obstacle_length", DEFAULTS["obstacle_length"])
    resolved_obstacle_clearance = resolve_param(
        obstacle_clearance, cfg, "obstacle_clearance", DEFAULTS["obstacle_clearance"]
    )
    resolved_obstacle_spacing = resolve_param(obstacle_spacing, cfg, "obstacle_spacing", DEFAULTS["obstacle_spacing"])
    resolved_obstacle_placement = resolve_param(
        obstacle_placement, cfg, "obstacle_placement", DEFAULTS["obstacle_placement"]
    )
    resolved_robot_radius = resolve_param(robot_radius, cfg, "robot_radius", DEFAULTS["robot_radius"])
    resolved_num_robots = resolve_param(num_robots, cfg, "num_robots", DEFAULTS["num_robots"])
    resolved_robot_min_clearance = resolve_param(
        robot_min_clearance, cfg, "robot_min_clearance", DEFAULTS["robot_min_clearance"]
    )
    resolved_spawn_export_filename = resolve_param(
        spawn_export_filename, cfg, "spawn_export_filename", DEFAULTS["spawn_export_filename"]
    )
    resolved_num_extra_points = resolve_param(
        num_extra_points, cfg, "num_extra_points", DEFAULTS["num_extra_points"]
    )
    resolved_extra_point_radius = resolve_param(
        extra_point_radius, cfg, "extra_point_radius", DEFAULTS["extra_point_radius"]
    )
    resolved_extra_point_min_clearance = resolve_param(
        extra_point_min_clearance, cfg, "extra_point_min_clearance", DEFAULTS["extra_point_min_clearance"]
    )

    # Validate required parameters
    if resolved_num_rooms is None:
        raise typer.BadParameter("num_rooms is required. Provide it via --num-rooms/-n or in a config file.")

    # Validate num_open_spaces constraint
    max_open_spaces = (resolved_num_turns + 1) // 2
    if resolved_num_open_spaces > max_open_spaces:
        raise typer.BadParameter(
            f"num_open_spaces ({resolved_num_open_spaces}) must be <= (num_turns + 1) / 2 "
            f"({max_open_spaces} with {resolved_num_turns} turns)"
        )

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
        obstacles_enabled=resolved_obstacles_enabled,
        num_obstacles=resolved_num_obstacles,
        obstacle_length=resolved_obstacle_length,
        obstacle_clearance=resolved_obstacle_clearance,
        obstacle_spacing=resolved_obstacle_spacing,
        obstacle_placement=resolved_obstacle_placement,
        robot_radius=resolved_robot_radius,
        num_robots=resolved_num_robots,
        robot_min_clearance=resolved_robot_min_clearance,
        spawn_export_filename=resolved_spawn_export_filename,
        num_extra_points=resolved_num_extra_points,
        extra_point_radius=resolved_extra_point_radius,
        extra_point_min_clearance=resolved_extra_point_min_clearance,
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
    if resolved_obstacles_enabled:
        typer.echo(f"  Obstacles: {resolved_num_obstacles}")
        typer.echo(f"  Obstacle size: {resolved_obstacle_length}m x {resolved_obstacle_length}m")
        typer.echo(f"  Obstacle clearance: {resolved_obstacle_clearance}m")
        typer.echo(f"  Obstacle spacing: {resolved_obstacle_spacing}m")
        typer.echo(f"  Obstacle placement: {resolved_obstacle_placement}")

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

    # Generate robot spawn positions and/or extra points if configured
    has_robots = params.robot_radius is not None
    has_extra_points = params.extra_point_radius is not None

    if has_robots or has_extra_points:
        from floorplan_generator.spawn import (
            generate_extra_point_positions,
            generate_spawn_positions,
            transform_to_map_center,
            write_spawn_yaml,
        )

        free_space = floorplan.get_free_space()
        bounds_for_center = floorplan.get_bounds()
        map_width = bounds_for_center[2] - bounds_for_center[0]
        map_height = bounds_for_center[3] - bounds_for_center[1]

        centered_robot_positions: list[tuple[float, float]] | None = None
        robot_positions_raw: list[tuple[float, float]] = []

        if has_robots:
            typer.echo(f"\nGenerating spawn positions for {params.num_robots} robots...")
            typer.echo(f"  Robot radius: {params.robot_radius}m")
            typer.echo(f"  Min clearance: {params.robot_min_clearance}m")

            robot_positions_raw = generate_spawn_positions(
                free_space=free_space,
                num_robots=params.num_robots,  # type: ignore[arg-type]
                robot_radius=params.robot_radius,
                min_clearance=params.robot_min_clearance,  # type: ignore[arg-type]
                resolution=resolved_resolution,
            )
            centered_robot_positions = transform_to_map_center(robot_positions_raw, bounds_for_center)

        centered_extra_points: list[tuple[float, float]] | None = None

        if has_extra_points:
            typer.echo(f"\nGenerating {params.num_extra_points} extra points...")
            typer.echo(f"  Extra point radius: {params.extra_point_radius}m")
            typer.echo(f"  Min clearance: {params.extra_point_min_clearance}m")

            extra_positions_raw = generate_extra_point_positions(
                free_space=free_space,
                num_extra_points=params.num_extra_points,  # type: ignore[arg-type]
                extra_point_radius=params.extra_point_radius,  # type: ignore[arg-type]
                min_clearance=params.extra_point_min_clearance,  # type: ignore[arg-type]
                existing_positions=robot_positions_raw,
                robot_radius=params.robot_radius or 0.0,
                resolution=resolved_resolution,
            )
            centered_extra_points = transform_to_map_center(extra_positions_raw, bounds_for_center)

        spawn_filename = params.spawn_export_filename or "world_config.yaml"
        spawn_path = output_path.parent / spawn_filename
        write_spawn_yaml(
            centered_robot_positions, map_width, map_height, spawn_path, extra_points=centered_extra_points
        )
        typer.echo(f"  Saved world config to {spawn_path}")

    # Print some stats
    bounds = floorplan.get_bounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    typer.echo(f"\nFloorplan dimensions: {width:.3f}m x {height:.3f}m")
    if floorplan.obstacles:
        typer.echo(f"Obstacles placed: {len(floorplan.obstacles)}/{params.num_obstacles}")

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
