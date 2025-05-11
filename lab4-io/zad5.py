#zad5
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Rasterizer:
    def __init__(self, width, height, scale=1):
        """
        Initialize a rasterizer with a black canvas.
        scale: Super sampling scale factor (default=1, no super sampling)
        """
        self.output_width = width
        self.output_height = height
        self.scale = scale

        # Create a working canvas in higher resolution for SSAA
        self.width = width * scale
        self.height = height * scale
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def clear(self, color=(0, 0, 0)):
        """Clear the canvas with the specified color."""
        self.canvas.fill(0)

    def draw_point(self, x, y, color=(255, 255, 255)):
        """Draw a single point on the canvas."""
        # Scale up the coordinates for the high-resolution canvas
        x_scaled = int(x * self.scale)
        y_scaled = int(y * self.scale)

        if 0 <= x_scaled < self.width and 0 <= y_scaled < self.height:
            # Ensure color is an array of uint8
            color_array = np.array(color, dtype=np.uint8)
            self.canvas[y_scaled, x_scaled] = color_array

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

    def draw_filled_triangle_with_color_interpolation(self, x1, y1, x2, y2, x3, y3, color1=(255, 0, 0),
                                                      color2=(0, 255, 0), color3=(0, 0, 255)):
        """
        Draw a filled triangle with color interpolation using barycentric coordinates.
        Uses the formula: Cp = (λ0/λ)*V0 + (λ1/λ)*V1 + (λ2/λ)*V2
        """
        # Calculate the bounding box for efficiency
        x_min = max(0, min(x1, x2, x3))
        x_max = min(self.output_width - 1, max(x1, x2, x3))
        y_min = max(0, min(y1, y2, y3))
        y_max = min(self.output_height - 1, max(y1, y2, y3))

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

    def create_output_image(self):
        """
        Create final output image by downsampling the high-resolution canvas.
        This performs the actual SSAA by averaging pixels from the working canvas.
        """
        if self.scale == 1:
            # No downsampling needed
            return Image.fromarray(self.canvas)

        # Create the output image array
        output = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # For each pixel in the output image, average the corresponding pixels in the high-res canvas
        for y in range(self.output_height):
            for x in range(self.output_width):
                # Calculate the region in the high-res canvas for this output pixel
                y_start = y * self.scale
                y_end = (y + 1) * self.scale
                x_start = x * self.scale
                x_end = (x + 1) * self.scale

                # Extract the region
                region = self.canvas[y_start:y_end, x_start:x_end]

                # Calculate the average color of the region
                avg_color = np.mean(region, axis=(0, 1))

                # Set the output pixel to the average color
                output[y, x] = avg_color.astype(np.uint8)

        return Image.fromarray(output)

    def save_image(self, filename):
        """Save the anti-aliased output canvas as an image file."""
        img = self.create_output_image()
        img.save(filename)

    def display(self):
        """Display the anti-aliased output canvas using matplotlib."""
        img = self.create_output_image()
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.show()


def main():
    # Create a 500x500 canvas with 2x super sampling
    scale = 2  # Super sampling scale factor
    raster = Rasterizer(500, 500, scale=scale)

    print("Please select an option:")
    print("1: View lines and triangles with color interpolation and anti-aliasing")
    print("2: Load and process an image from your computer")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
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

        # Display the result (anti-aliased)
        raster.display()

        # Save the image
        raster.save_image("antialiased_color_demo.png")
        print("Demo image saved as 'antialiased_color_demo.png'")

    elif choice == "2":
        try:
            # Get image path from user
            image_path = input("Enter the path to your image file: ")

            # Load the image
            input_image = Image.open(image_path)

            # Resize if needed to fit our canvas
            if input_image.width > raster.output_width or input_image.height > raster.output_height:
                input_image = input_image.resize((raster.output_width, raster.output_height))

            # Convert to numpy array
            img_array = np.array(input_image)

            # Create a new rasterizer with the image dimensions
            img_raster = Rasterizer(input_image.width, input_image.height, scale=scale)

            # Apply some rasterization effects to the image
            # For example, we could draw triangles over it or extract edges
            # For simplicity, let's just copy the image to our canvas and apply anti-aliasing

            # First scale up the image to our high-res canvas
            for y in range(img_array.shape[0]):
                for x in range(img_array.shape[1]):
                    # Get pixel color
                    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                        color = (img_array[y, x, 0], img_array[y, x, 1], img_array[y, x, 2])
                    else:
                        # Grayscale image
                        color = (img_array[y, x], img_array[y, x], img_array[y, x])

                    # Draw on our high-res canvas
                    img_raster.draw_point(x, y, color)

            # Display the anti-aliased result
            img_raster.display()

            # Save the output
            output_path = f"antialiased_{image_path.split('/')[-1]}"
            img_raster.save_image(output_path)
            print(f"Processed image saved as '{output_path}'")

        except Exception as e:
            print(f"Error processing image: {e}")
            print("Falling back to demo mode...")
            # Run the demo if there's an error with the image
            main()  # Restart

    else:
        print("Invalid choice. Please enter 1 or 2.")
        main()  # Restart
