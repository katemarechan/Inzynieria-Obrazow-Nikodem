#zad4
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Rasterizer:
    def __init__(self, width, height):
        """Initialize a rasterizer with a black canvas."""
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def clear(self, color=(0, 0, 0)):
        """Clear the canvas with the specified color."""
        self.canvas.fill(0)

    def draw_point(self, x, y, color=(255, 255, 255)):
        """Draw a single point on the canvas."""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Ensure color is an array of uint8
            color_array = np.array(color, dtype=np.uint8)
            self.canvas[int(y), int(x)] = color_array

    def interpolate_color(self, color1, color2, t):
        """
        Interpolate between two colors using the formula: C = A + t * (B - A)
        where t is in [0, 1] range
        """
        return tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(color1, color2))

    def draw_line_with_color_interpolation(self, x1, y1, x2, y2, color1=(255, 255, 255), color2=(255, 255, 255)):
        """Draw a line with color interpolation using Bresenham's algorithm."""
        # Calculate differences and determine direction
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        # Determine dominant direction
        is_x_dominant = dx > dy

        # Total steps for normalization
        total_steps = dx if is_x_dominant else dy
        if total_steps == 0:  # Avoid division by zero for single point
            self.draw_point(x1, y1, color1)
            return

        # Initialize error term
        if is_x_dominant:
            err = 2 * dy - dx
        else:
            err = 2 * dx - dy

        # Starting point
        x, y = x1, y1

        # Keep track of steps for color interpolation
        steps_taken = 0

        # Bresenham main loop
        while True:
            # Calculate t for color interpolation (progress along the line)
            t = steps_taken / total_steps

            # Interpolate color using the formula C = A + t * (B - A)
            color = self.interpolate_color(color1, color2, t)

            # Draw the point with interpolated color
            self.draw_point(x, y, color)

            # Check if we've reached the end point
            if x == x2 and y == y2:
                break

            # Update coordinates based on dominant direction
            if is_x_dominant:
                # Move in X direction
                x += sx
                # Check if we need to move in Y direction
                if err >= 0:
                    y += sy
                    err -= 2 * dx
                err += 2 * dy
            else:
                # Move in Y direction
                y += sy
                # Check if we need to move in X direction
                if err >= 0:
                    x += sx
                    err -= 2 * dy
                err += 2 * dx

            steps_taken += 1

    def calculate_area(self, x1, y1, x2, y2, x3, y3):
        """Calculate area of a triangle using cross product (2D)."""
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def is_point_in_triangle(self, x, y, x1, y1, x2, y2, x3, y3):
        """Check if a point (x,y) is inside a triangle using barycentric coordinates."""
        # Calculate the total area of the triangle
        total_area = self.calculate_area(x1, y1, x2, y2, x3, y3)

        # Return early if triangle is degenerate (area = 0)
        if total_area == 0:
            return False

        # Calculate areas of three triangles formed by the point and two vertices
        area1 = self.calculate_area(x, y, x2, y2, x3, y3)  # Area(P, V1, V2)
        area2 = self.calculate_area(x1, y1, x, y, x3, y3)  # Area(V0, P, V2)
        area3 = self.calculate_area(x1, y1, x2, y2, x, y)  # Area(V0, V1, P)

        # Check if the sum of areas is equal to the total area (with some epsilon for floating point)
        return abs(total_area - (area1 + area2 + area3)) < 1e-9

    def draw_filled_triangle_with_color_interpolation(self, x1, y1, x2, y2, x3, y3, color1=(255, 0, 0),
                                                      color2=(0, 255, 0), color3=(0, 0, 255)):
        """
        Draw a filled triangle with color interpolation using barycentric coordinates.
        Uses the formula: Cp = (λ0/λ)*V0 + (λ1/λ)*V1 + (λ2/λ)*V2
        where:
        λ = Area(V0, V1, V2)
        λ0 = Area(V0, V1, P)
        λ1 = Area(V1, V2, P)
        λ2 = Area(V2, V0, P)
        """
        # Calculate the bounding box for efficiency
        x_min = max(0, min(x1, x2, x3))
        x_max = min(self.width - 1, max(x1, x2, x3))
        y_min = max(0, min(y1, y2, y3))
        y_max = min(self.height - 1, max(y1, y2, y3))

        # Calculate the total area of the triangle
        total_area = self.calculate_area(x1, y1, x2, y2, x3, y3)

        # Return early if triangle is degenerate (area = 0)
        if total_area == 0:
            return

        # Iterate through points in the bounding box
        for y in range(int(y_min), int(y_max) + 1):
            for x in range(int(x_min), int(x_max) + 1):
                # Calculate barycentric coordinates
                lambda0 = self.calculate_area(x2, y2, x3, y3, x, y)  # Area(V1, V2, P)
                lambda1 = self.calculate_area(x3, y3, x1, y1, x, y)  # Area(V2, V0, P)
                lambda2 = self.calculate_area(x1, y1, x2, y2, x, y)  # Area(V0, V1, P)

                # Check if point is inside the triangle
                if abs(total_area - (lambda0 + lambda1 + lambda2)) < 1e-9:
                    # Normalize the barycentric coordinates
                    lambda0 /= total_area
                    lambda1 /= total_area
                    lambda2 /= total_area

                    # Interpolate color using barycentric coordinates
                    # C = (λ0/λ)*V0 + (λ1/λ)*V1 + (λ2/λ)*V2
                    r = int(lambda0 * color1[0] + lambda1 * color2[0] + lambda2 * color3[0])
                    g = int(lambda0 * color1[1] + lambda1 * color2[1] + lambda2 * color3[1])
                    b = int(lambda0 * color1[2] + lambda1 * color2[2] + lambda2 * color3[2])

                    # Ensure color values are in valid range
                    color = (
                        max(0, min(255, r)),
                        max(0, min(255, g)),
                        max(0, min(255, b))
                    )

                    # Draw the point with interpolated color
                    self.draw_point(x, y, color)

    def draw_triangle_outline_with_color_interpolation(self, x1, y1, x2, y2, x3, y3, color1=(255, 0, 0),
                                                       color2=(0, 255, 0), color3=(0, 0, 255)):
        """Draw the outline of a triangle with color interpolation."""
        self.draw_line_with_color_interpolation(x1, y1, x2, y2, color1, color2)
        self.draw_line_with_color_interpolation(x2, y2, x3, y3, color2, color3)
        self.draw_line_with_color_interpolation(x3, y3, x1, y1, color3, color1)

    def save_image(self, filename):
        """Save the canvas as an image file."""
        img = Image.fromarray(self.canvas)
        img.save(filename)

    def display(self):
        """Display the canvas using matplotlib."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.canvas)
        plt.axis('off')
        plt.show()


def main():
    # Create a 500x500 canvas
    raster = Rasterizer(500, 500)

    # Demo color interpolation with lines
    # Line with red to blue gradient
    raster.draw_line_with_color_interpolation(50, 50, 450, 100,
                                              color1=(255, 0, 0),  # Red
                                              color2=(0, 0, 255))  # Blue

    # Line with green to yellow gradient
    raster.draw_line_with_color_interpolation(50, 150, 450, 200,
                                              color1=(0, 255, 0),  # Green
                                              color2=(255, 255, 0))  # Yellow

    # Demo color interpolation with triangles
    # Triangle outline with red, green, blue vertices
    raster.draw_triangle_outline_with_color_interpolation(
        100, 300, 250, 150, 400, 350,
        color1=(255, 0, 0),  # Red
        color2=(0, 255, 0),  # Green
        color3=(0, 0, 255)  # Blue
    )

    # Filled triangle with RGB color vertices
    raster.draw_filled_triangle_with_color_interpolation(
        200, 400, 350, 250, 450, 450,
        color1=(255, 0, 0),  # Red
        color2=(0, 255, 0),  # Green
        color3=(0, 0, 255)  # Blue
    )

    # Display the result
    raster.display()

    # Save the image
    raster.save_image("color_interpolation_demo.png")
