from PIL import Image
import os


class PPMHandler:
    def __init__(self):
        pass

    def image_to_p3_ppm(self, image_path, output_path):

        # Open the image using Pillow and convert to RGB
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        # Create the P3 PPM file
        with open(output_path, 'w') as f:
            # Write the header
            f.write(f"P3\n{width} {height}\n255\n")

            # Write the pixel data
            pixels = list(img.getdata())
            pixel_count = 0

            for pixel in pixels:
                f.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
                pixel_count += 1

                # Add a newline every 5 pixels for readability
                if pixel_count % 5 == 0:
                    f.write('\n')

        print(f"Image saved as P3 PPM format to {output_path}")

    def image_to_p6_ppm(self, image_path, output_path):

        # Open the image using Pillow and convert to RGB
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        # Create the P6 PPM file
        with open(output_path, 'wb') as f:
            # Write the header (as bytes)
            f.write(f"P6\n{width} {height}\n255\n".encode())

            # Write the pixel data directly as binary
            f.write(img.tobytes())

        print(f"Image saved as P6 PPM format to {output_path}")

    def read_ppm(self, ppm_path):

        # Determine if it's P3 or P6 by checking the first few bytes
        with open(ppm_path, 'rb') as f:
            header = f.readline().decode().strip()

        if header == "P3":
            return self._read_p3_ppm(ppm_path)
        elif header == "P6":
            return self._read_p6_ppm(ppm_path)
        else:
            raise ValueError(f"Unsupported PPM format: {header}")

    def _read_p3_ppm(self, ppm_path):

        with open(ppm_path, 'r') as f:
            # Read the header
            magic = f.readline().strip()
            if magic != "P3":
                raise ValueError("Not a P3 PPM file")

            # Skip any comment lines
            line = f.readline()
            while line.startswith('#'):
                line = f.readline()

            # Read dimensions
            width, height = map(int, line.split())

            # Read max value
            max_val = int(f.readline().strip())
            if max_val != 255:
                raise ValueError(f"Unsupported max value: {max_val}")

            # Read pixel data
            data = []
            for line in f:
                if line.strip() and not line.startswith('#'):
                    data.extend([int(x) for x in line.split()])

            # Convert to image
            pixels = []
            for i in range(0, len(data), 3):
                if i + 2 < len(data):  # Ensure we have 3 values for RGB
                    pixels.append((data[i], data[i + 1], data[i + 2]))

            img = Image.new('RGB', (width, height))
            img.putdata(pixels)
            return img

    def _read_p6_ppm(self, ppm_path):

        with open(ppm_path, 'rb') as f:
            # Read and validate the header
            magic = f.readline().decode().strip()
            if magic != "P6":
                raise ValueError("Not a P6 PPM file")

            # Skip any comment lines
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()

            # Read dimensions
            width, height = map(int, line.decode().split())

            # Read max value
            max_val = int(f.readline().decode().strip())
            if max_val != 255:
                raise ValueError(f"Unsupported max value: {max_val}")

            # Read pixel data
            raw_data = f.read()
            img = Image.frombytes('RGB', (width, height), raw_data)
            return img

    def compare_file_sizes(self, p3_path, p6_path):

        p3_size = os.path.getsize(p3_path)
        p6_size = os.path.getsize(p6_path)

        ratio = p3_size / p6_size if p6_size > 0 else 0

        print(f"P3 (text) file size: {p3_size} bytes")
        print(f"P6 (binary) file size: {p6_size} bytes")
        print(f"P3 is {ratio:.2f} times larger than P6")

        return (p3_size, p6_size, ratio)

    def print_file_structure(self, file_path, is_binary=False, max_lines=10):

        print(f"\nStructure of {file_path}:")

        if is_binary:
            with open(file_path, 'rb') as f:
                # Print header (usually text even in binary files)
                header_lines = 0
                line = f.readline()
                while header_lines < 3:  # PPM has 3 header lines
                    print(line.decode().strip())
                    line = f.readline()
                    header_lines += 1

                # Print some of the binary data in hex representation
                print("Binary data (showing first 50 bytes in hex):")
                binary_data = f.read(50)
                hex_representation = ' '.join([f'{b:02x}' for b in binary_data])
                print(hex_representation)
        else:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()[:max_lines]]
                for line in lines:
                    print(line)
                if max_lines < sum(1 for _ in open(file_path, 'r')):
                    print("...")


def demo():

    handler = PPMHandler()

    # Ask the user for the input file name
    photo_path = input("Enter the name of the image file you want to transform: ")

    # Check if the file exists
    if not os.path.exists(photo_path):
        print(f"Error: The file '{photo_path}' does not exist. Please ensure the file is in the current directory.")
        return

    # Get the file name without extension to create output file names
    file_base = os.path.splitext(photo_path)[0]
    p3_output = f"{file_base}_p3.ppm"
    p6_output = f"{file_base}_p6.ppm"

    print(f"\n===== Processing Image '{photo_path}' as P3 and P6 PPM =====")

    # Convert the image to P3 and P6 formats
    handler.image_to_p3_ppm(photo_path, p3_output)
    handler.image_to_p6_ppm(photo_path, p6_output)

    # Show the structure of the saved files
    handler.print_file_structure(p3_output)
    handler.print_file_structure(p6_output, is_binary=True)

    # Read back the PPM files and save as PNG for verification
    p3_img = handler.read_ppm(p3_output)
    p3_read_output = f"{file_base}_p3_read.png"
    p3_img.save(p3_read_output)
    print(f"P3 image read and saved as {p3_read_output}")

    p6_img = handler.read_ppm(p6_output)
    p6_read_output = f"{file_base}_p6_read.png"
    p6_img.save(p6_read_output)
    print(f"P6 image read and saved as {p6_read_output}")

    # Compare file sizes
    handler.compare_file_sizes(p3_output, p6_output)


if __name__ == "__main__":
    demo()