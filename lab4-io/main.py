import os
import sys
import importlib.util


def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def import_module_from_file(file_path):
    """Import a module from file path."""
    module_name = os.path.basename(file_path).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def display_menu():
    """Display the main menu."""
    clear_screen()
    print("=" * 50)
    print("Image Processing and Rasterization Toolkit")
    print("=" * 50)
    print("1. Grayscale Floyd-Steinberg Dithering")
    print("2. Color Floyd-Steinberg Dithering")
    print("3. Basic Rasterization (Lines and Triangles)")
    print("4. Color Interpolation Rasterization")
    print("5. Super-sampling Anti-aliasing (SSAA)")
    print("0. Exit")
    print("=" * 50)


def run_module(file_name):
    """Run a specific module."""
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, file_name)

        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' not found.")
            input("Press Enter to continue...")
            return

        # Import and run the module
        module = import_module_from_file(file_path)

        # Execute the main function of the module
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"Error: No main() function found in {file_name}")

        input("\nTask completed. Press Enter to return to the menu...")

    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press Enter to continue...")


def main():
    """Main program loop."""
    while True:
        display_menu()
        choice = input("Enter your choice (0-5): ")

        if choice == '0':
            print("Exiting program. Goodbye!")
            sys.exit(0)
        elif choice == '1':
            run_module("zad1.py")
        elif choice == '2':
            run_module("zad2.py")
        elif choice == '3':
            run_module("zad3.py")
        elif choice == '4':
            run_module("zad4.py")
        elif choice == '5':
            run_module("zad5.py")
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()