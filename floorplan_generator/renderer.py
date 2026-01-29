"""PNG rendering for floorplans."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LinearRing, MultiPolygon, Polygon

from floorplan_generator.generator import Floorplan


class FloorplanRenderer:
    """Renders floorplans to PNG occupancy grids."""

    def __init__(self, resolution: float = 0.05, padding: float = 1.0) -> None:
        """
        Initialize the renderer.

        Args:
            resolution: Meters per pixel (e.g., 0.05 = 20 pixels per meter).
            padding: Padding around the floorplan in meters.

        """
        self.resolution = resolution
        self.padding = padding

    def render_to_png(self, floorplan: Floorplan, output_path: str | Path) -> None:
        """
        Render the floorplan to a PNG file.

        Args:
            floorplan: The floorplan to render.
            output_path: Path to save the PNG file.

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get bounds and calculate image size
        min_x, min_y, max_x, max_y = floorplan.get_bounds()
        min_x -= self.padding
        min_y -= self.padding
        max_x += self.padding
        max_y += self.padding

        width_m = max_x - min_x
        height_m = max_y - min_y

        width_px = int(np.ceil(width_m / self.resolution))
        height_px = int(np.ceil(height_m / self.resolution))

        # Create image with white background (free space outside floorplan)
        # For Stage: white = free, black = obstacle
        image = Image.new("L", (width_px, height_px), color=255)
        draw = ImageDraw.Draw(image)

        # Helper to convert world coordinates to pixel coordinates
        def world_to_pixel(x: float, y: float) -> tuple[int, int]:
            px = round((x - min_x) / self.resolution)
            # Flip y-axis (image origin is top-left, world origin is bottom-left)
            py = height_px - 1 - round((y - min_y) / self.resolution)
            return (px, py)

        def ring_to_pixel_coords(ring: LinearRing) -> list[tuple[int, int]]:
            """Convert a LinearRing to pixel coordinates."""
            return [world_to_pixel(x, y) for x, y in ring.coords]

        def draw_polygon_with_holes(polygon: Polygon, fill: int, hole_fill: int) -> None:
            """Draw a polygon, properly handling interior holes."""
            # Draw the exterior
            exterior_coords = ring_to_pixel_coords(polygon.exterior)
            draw.polygon(exterior_coords, fill=fill)
            # Draw holes with the background/hole color
            for interior in polygon.interiors:
                hole_coords = ring_to_pixel_coords(interior)
                draw.polygon(hole_coords, fill=hole_fill)

        # Draw walls as black, with holes filled as white (background)
        if isinstance(floorplan.walls, MultiPolygon):
            for geom in floorplan.walls.geoms:
                if isinstance(geom, Polygon):
                    draw_polygon_with_holes(geom, fill=0, hole_fill=255)
        elif isinstance(floorplan.walls, Polygon):
            draw_polygon_with_holes(floorplan.walls, fill=0, hole_fill=255)

        # Draw free space (rooms and hallway interiors) as white
        # This ensures the interior is white even if walls overlap
        free_space = floorplan.get_free_space()
        if isinstance(free_space, MultiPolygon):
            for geom in free_space.geoms:
                if isinstance(geom, Polygon):
                    exterior_coords = ring_to_pixel_coords(geom.exterior)
                    draw.polygon(exterior_coords, fill=255)
        elif isinstance(free_space, Polygon):
            exterior_coords = ring_to_pixel_coords(free_space.exterior)
            draw.polygon(exterior_coords, fill=255)

        # Save the image
        image.save(output_path)

    def render_debug(self, floorplan: Floorplan, output_path: str | Path) -> None:
        """
        Render a debug visualization with colored regions.

        Args:
            floorplan: The floorplan to render.
            output_path: Path to save the PNG file.

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get bounds and calculate image size
        min_x, min_y, max_x, max_y = floorplan.get_bounds()
        min_x -= self.padding
        min_y -= self.padding
        max_x += self.padding
        max_y += self.padding

        width_m = max_x - min_x
        height_m = max_y - min_y

        width_px = int(np.ceil(width_m / self.resolution))
        height_px = int(np.ceil(height_m / self.resolution))

        # Create RGB image
        image = Image.new("RGB", (width_px, height_px), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)

        def world_to_pixel(x: float, y: float) -> tuple[int, int]:
            px = round((x - min_x) / self.resolution)
            py = height_px - 1 - round((y - min_y) / self.resolution)
            return (px, py)

        def ring_to_pixel_coords(ring: LinearRing) -> list[tuple[int, int]]:
            """Convert a LinearRing to pixel coordinates."""
            return [world_to_pixel(x, y) for x, y in ring.coords]

        def draw_polygon_with_holes(
            polygon: Polygon,
            fill: tuple[int, int, int],
            hole_fill: tuple[int, int, int],
        ) -> None:
            """Draw a polygon, properly handling interior holes."""
            exterior_coords = ring_to_pixel_coords(polygon.exterior)
            draw.polygon(exterior_coords, fill=fill)
            for interior in polygon.interiors:
                hole_coords = ring_to_pixel_coords(interior)
                draw.polygon(hole_coords, fill=hole_fill)

        # Draw walls first (dark gray), with holes as background color
        bg_color = (240, 240, 240)
        if isinstance(floorplan.walls, MultiPolygon):
            for geom in floorplan.walls.geoms:
                if isinstance(geom, Polygon):
                    draw_polygon_with_holes(geom, fill=(60, 60, 60), hole_fill=bg_color)
        elif isinstance(floorplan.walls, Polygon):
            draw_polygon_with_holes(floorplan.walls, fill=(60, 60, 60), hole_fill=bg_color)

        # Draw hallway interior (light blue)
        if isinstance(floorplan.hallway_interior, Polygon):
            exterior_coords = ring_to_pixel_coords(floorplan.hallway_interior.exterior)
            draw.polygon(exterior_coords, fill=(200, 220, 255))

        # Draw room interiors (light green)
        for room in floorplan.room_interiors:
            if isinstance(room, Polygon):
                exterior_coords = ring_to_pixel_coords(room.exterior)
                draw.polygon(exterior_coords, fill=(200, 255, 200))

        # Draw doors (orange) - these connect rooms to hallway
        for door in floorplan.doors:
            if isinstance(door, Polygon):
                exterior_coords = ring_to_pixel_coords(door.exterior)
                draw.polygon(exterior_coords, fill=(255, 165, 0))

        # Save the image
        image.save(output_path)
