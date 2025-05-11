import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RGBCubeVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.bg_color = (235, 235, 235)

        # Colors for the corners of the RGB cube (matches the provided diagram)
        self.corner_colors = {
            0: (0, 0, 0),  # Black (origin)
            1: (0, 0, 255),  # Blue
            2: (0, 255, 0),  # Green
            3: (0, 255, 255),  # Cyan
            4: (255, 0, 0),  # Red
            5: (255, 0, 255),  # Magenta
            6: (255, 255, 0),  # Yellow
            7: (255, 255, 255)  # White
        }

        # Transition paths as shown in the diagram
        self.path_1_to_7 = [0, 1, 3, 7]  # Black → Blue → Cyan → White
        self.path_1_to_8 = [0, 4, 6, 7]  # Black → Red → Yellow → White

    def _interpolate_color(self, color1, color2, ratio):
        """Interpolate between two colors"""
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        return (r, g, b)

    def create_spectrum(self, path, width=800, height=100):
        """Create a color spectrum image for a given path"""
        # Create an array for the spectrum
        spectrum = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate how many pixels per segment
        segment_width = width // (len(path) - 1)

        # Fill the spectrum
        for i in range(len(path) - 1):
            start_color = self.corner_colors[path[i]]
            end_color = self.corner_colors[path[i + 1]]

            for j in range(segment_width):
                ratio = j / segment_width
                color = self._interpolate_color(start_color, end_color, ratio)

                x = i * segment_width + j
                if x < width:  # Ensure we don't go out of bounds
                    spectrum[:, x] = color

        # Add markers for the vertices
        marker_positions = [i * segment_width for i in range(len(path))]
        for i, pos in enumerate(marker_positions):
            if pos < width:
                # Draw a small vertical marker line at each vertex point
                marker_height = 10
                spectrum[height - marker_height:height, pos] = (0, 0, 0)

        return spectrum

    def save_ppm(self, array, filename, format='P3'):
        """Save numpy array as PPM file"""
        height, width, _ = array.shape

        with open(filename, 'w') as f:
            # Write PPM header
            f.write(f"P3\n{width} {height}\n255\n")

            # Write pixel data
            for y in range(height):
                for x in range(width):
                    r, g, b = array[y, x]
                    f.write(f"{r} {g} {b} ")
                f.write("\n")

        print(f"Saved {filename}")

        # Also save as PNG
        png_filename = filename.replace('.ppm', '.png')
        img = Image.fromarray(array.astype('uint8'))
        img.save(png_filename)
        print(f"Saved {png_filename}")

    def generate_visualization(self):
        """Generate all required visualizations"""
        # Create spectrum for path 1 to 7
        spectrum1 = self.create_spectrum(self.path_1_to_7)
        self.save_ppm(spectrum1, "spectrum_1_to_7.ppm")

        # Create additional spectrum for path 1 to 8 (0-4-6-7)
        spectrum2 = self.create_spectrum(self.path_1_to_8)
        self.save_ppm(spectrum2, "spectrum_1_to_8.ppm")

        # Combine both spectrums with labels into one image
        combined_height = 250
        combined = np.zeros((combined_height, self.width, 3), dtype=np.uint8)
        combined.fill(235)  # Fill with background color

        # Add title text area
        title_height = 50
        spectrum_height = 100

        # Add spectrums
        combined[title_height:title_height + spectrum_height, :spectrum1.shape[1]] = spectrum1
        combined[title_height + spectrum_height:title_height + 2 * spectrum_height, :spectrum2.shape[1]] = spectrum2

        # Save combined image
        self.save_ppm(combined, "rgb_visualization.ppm")

        # Create a more detailed PNG with labels using PIL
        img = Image.fromarray(combined.astype('uint8'))
        draw = ImageDraw.Draw(img)

        # Try to use a font if available, otherwise PIL will use a default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Add labels
        draw.text((10, 15), "RGB Cube Color Transitions", fill=(0, 0, 0), font=font)
        draw.text((10, title_height - 20), "Path 0→1→3→7: Black → Blue → Cyan → White", fill=(0, 0, 0), font=font)
        draw.text((10, title_height + spectrum_height - 20), "Path 0→4→6→7: Black → Red → Yellow → White",
                  fill=(0, 0, 0), font=font)

        # Save the enhanced PNG
        img.save("rgb_visualization_with_labels.png")
        print("Saved rgb_visualization_with_labels.png")


# Create and run the visualizer
visualizer = RGBCubeVisualizer()
visualizer.generate_visualization()