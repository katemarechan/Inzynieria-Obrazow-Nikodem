import os
import struct
import zlib
import numpy as np



def create_rgb_spectrum(width=800, height=100):
    """Create the RGB spectrum from previous task"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Define colors for our spectrum (based on previous task)
    colors = [
        (0, 0, 255),  # Blue
        (0, 255, 255),  # Cyan
        (0, 255, 0),  # Green
        (255, 255, 0),  # Yellow
        (255, 0, 0),  # Red
        (255, 0, 255),  # Magenta
        (255, 255, 255)  # White
    ]

    # Calculate segment width
    segment_width = width // (len(colors) - 1)

    # Fill the image with interpolated colors
    for i in range(len(colors) - 1):
        c1 = np.array(colors[i])
        c2 = np.array(colors[i + 1])

        start_x = i * segment_width
        end_x = (i + 1) * segment_width

        for x in range(start_x, end_x):
            # Calculate interpolation ratio
            t = (x - start_x) / segment_width
            # Linear interpolation
            color = (1 - t) * c1 + t * c2
            # Fill column with this color
            img[:, x] = color.astype(np.uint8)

    # Handle any remaining pixels
    if width % segment_width != 0:
        img[:, (len(colors) - 1) * segment_width:] = colors[-1]

    return img


def create_custom_png(image_data, output_filename):
    """
    Create a custom PNG file according to the assignment specifications.

    Args:
        image_data: NumPy array of RGB image data
        output_filename: Name of the output PNG file
    """
    height, width, _ = image_data.shape

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk (header)
    # From the task: width and height (2x uint32), bit depth, color model - numbers 8, 2, 0, 0 (5x uint8)
    ihdr_data = struct.pack('>IIBBBBB',
                            width,  # Width (uint32)
                            height,  # Height (uint32)
                            8,  # Bit depth (uint8) - 8 bits per channel
                            2,  # Color type (uint8) - 2 for RGB
                            0,  # Compression method (uint8) - 0 for zlib/deflate
                            0,  # Filter method (uint8) - 0 for standard PNG filtering
                            0)  # Interlace method (uint8) - 0 for no interlacing

    ihdr_chunk = create_chunk(b'IHDR', ihdr_data)

    # Prepare image data
    raw_data = b''
    for y in range(height):
        # Add filter byte (0) at the beginning of each scanline
        raw_data += b'\x00'
        for x in range(width):
            r, g, b = image_data[y, x]
            raw_data += struct.pack('BBB', r, g, b)

    # Compress image data using zlib as specified in the task
    compressed_data = zlib.compress(raw_data)

    # IDAT chunk (image data)
    idat_chunk = create_chunk(b'IDAT', compressed_data)

    # IEND chunk (end of PNG)
    iend_chunk = create_chunk(b'IEND', b'')

    # Write PNG file
    with open(output_filename, 'wb') as f:
        f.write(signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)

    print(f"PNG file created: {output_filename}")

    # Get file size
    file_size = os.path.getsize(output_filename)
    print(f"File size: {file_size} bytes")


def create_chunk(chunk_type, chunk_data):
    """
    Create a PNG chunk with the given type and data.

    Args:
        chunk_type: 4-byte chunk type
        chunk_data: chunk data bytes

    Returns:
        Bytes representing the full chunk (length + type + data + CRC)
    """
    # Length (4 bytes) + Type (4 bytes) + Data + CRC (4 bytes)
    length = len(chunk_data)

    # Calculate CRC (type + data)
    crc_input = chunk_type + chunk_data
    crc = zlib.crc32(crc_input) & 0xFFFFFFFF

    # Construct chunk
    chunk = struct.pack('>I', length) + chunk_type + chunk_data + struct.pack('>I', crc)

    return chunk


def main():
    # Create RGB spectrum from previous task
    print("Creating RGB spectrum image...")
    spectrum = create_rgb_spectrum(width=800, height=100)

    # Create custom PNG file with specified header and zlib compression
    print("Creating custom PNG file with specifications from the task...")
    create_custom_png(spectrum, 'rgb_spectrum_task3.png')

    # Print information about the PNG structure
    print("\nPNG file structure:")
    print("1. PNG signature (8 bytes)")
    print("2. IHDR chunk:")
    print("   - Length: 13 bytes")
    print("   - Type: 'IHDR'")
    print("   - Data: Width (4 bytes), Height (4 bytes), Bit depth (1 byte),")
    print("           Color type (1 byte), Compression (1 byte), Filter (1 byte),")
    print("           Interlace (1 byte)")
    print("   - CRC: 4 bytes")
    print("3. IDAT chunk:")
    print("   - Length: Variable")
    print("   - Type: 'IDAT'")
    print("   - Data: zlib-compressed image data")
    print("   - CRC: 4 bytes")
    print("4. IEND chunk:")
    print("   - Length: 0 bytes")
    print("   - Type: 'IEND'")
    print("   - Data: (empty)")
    print("   - CRC: 4 bytes")


if __name__ == "__main__":
    main()