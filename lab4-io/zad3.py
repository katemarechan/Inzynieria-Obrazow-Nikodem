#zad3
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
            self.canvas[y, x] = color

    def draw_line_bresenham(self, x1, y1, x2, y2, color=(255, 255, 255)):
        """Draw a line using Bresenham's algorithm."""
        # Calculate differences and determine direction
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        # Determine dominant direction (which axis has larger change)
        is_x_dominant = dx > dy

        # Initialize error term
        if is_x_dominant:
            err = 2 * dy - dx
        else:
            err = 2 * dx - dy

        # Starting point
        x, y = x1, y1

        # Bresenham main loop
        while True:
            # Draw the point
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

    def calculate_area(self, x1, y1, x2, y2, x3, y3):
        """Calculate area of a triangle using cross product (2D)."""
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def is_point_in_triangle(self, x, y, x1, y1, x2, y2, x3, y3):
        """
        Check if a point (x,y) is inside a triangle using barycentric coordinates
        based on calculating areas as mentioned in slide 12.
        """
        # Calculate the total area of the triangle
        total_area = self.calculate_area(x1, y1, x2, y2, x3, y3)

        # Calculate areas of three triangles formed by the point and two vertices
        area1 = self.calculate_area(x, y, x2, y2, x3, y3)
        area2 = self.calculate_area(x1, y1, x, y, x3, y3)
        area3 = self.calculate_area(x1, y1, x2, y2, x, y)

        # Check if the sum of areas is equal to the total area (with some epsilon for floating point)
        return abs(total_area - (area1 + area2 + area3)) < 1e-9

    def draw_filled_triangle(self, x1, y1, x2, y2, x3, y3, color=(255, 255, 255)):
        """
        Draw a filled triangle by determining which points lie inside the triangle.
        Uses a bounding box to limit the number of points that need to be tested.
        """
        # Calculate the bounding box for efficiency
        x_min = max(0, min(x1, x2, x3))
        x_max = min(self.width - 1, max(x1, x2, x3))
        y_min = max(0, min(y1, y2, y3))
        y_max = min(self.height - 1, max(y1, y2, y3))

        # Iterate through points in the bounding box
        for y in range(int(y_min), int(y_max) + 1):
            for x in range(int(x_min), int(x_max) + 1):
                # Check if the current point is inside the triangle
                if self.is_point_in_triangle(x, y, x1, y1, x2, y2, x3, y3):
                    self.draw_point(x, y, color)

    def draw_triangle_outline(self, x1, y1, x2, y2, x3, y3, color=(255, 255, 255)):
        """Draw just the outline of a triangle using Bresenham's line algorithm."""
        self.draw_line_bresenham(x1, y1, x2, y2, color)
        self.draw_line_bresenham(x2, y2, x3, y3, color)
        self.draw_line_bresenham(x3, y3, x1, y1, color)

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
    # Create a 400x300 canvas
    raster = Rasterizer(400, 300)

    # Demo drawing lines
    raster.draw_line_bresenham(50, 50, 200, 100, color=(255, 0, 0))  # Red line
    raster.draw_line_bresenham(50, 150, 350, 50, color=(0, 255, 0))  # Green line

    # Demo drawing triangles
    # Outline triangle
    raster.draw_triangle_outline(100, 200, 200, 100, 300, 250, color=(0, 0, 255))  # Blue outline

    # Filled triangle
    raster.draw_filled_triangle(200, 200, 300, 150, 250, 250, color=(255, 255, 0))  # Yellow filled triangle

    # Display the result
    raster.display()

    # Optional: save the image
    raster.save_image("rasterization_demo.png")

