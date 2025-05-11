import os
import sys
import importlib.util


def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_menu():
    """Displays the menu for task selection."""
    print("\n==== Image Processing Tasks ====")
    print("1. PPM Format Handling (Task 1)")
    print("2. RGB Cube Visualization (Task 2)")
    print("3. Custom PNG Creation (Task 3)")
    print("4. JPEG Algorithm Implementation (Task 4)")
    print("0. Exit")
    print("===============================")


def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    try:
        # First try standard import
        if module_name in sys.modules:
            return sys.modules[module_name]

        try:
            # Try direct import first (works when not in PyInstaller bundle)
            return importlib.import_module(module_name)
        except ImportError:
            # If direct import fails, try to load from file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Module {module_name} not found at {file_path}")

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Add to sys.modules
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        raise


def execute_task(task_number):
    """Executes a task based on the provided task number."""
    try:
        # Determine the base path for the task files
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        print(f"Base path: {base_path}")  # Debugging line

        # Construct the path to the task file
        module_name = f"zad{task_number}"
        module_path = os.path.join(base_path, f"{module_name}.py")

        # Check if the task file exists and import it
        if os.path.exists(module_path):
            module = import_module_from_file(module_name, module_path)

            # Try to call the main() function or other entry points
            if hasattr(module, 'main') and callable(module.main):
                module.main()
            elif hasattr(module, 'demo') and callable(module.demo):
                module.demo()
            else:
                # Fallback: Look for class with generate_visualization method
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, 'generate_visualization'):
                        instance = attr()
                        instance.generate_visualization()
                        break
                else:
                    print(f"No known entry point found in Task {task_number}.")
        else:
            print(f"Task file zad{task_number}.py not found at path: {module_path}")

        input("\nTask execution completed. Press Enter to continue...")

    except Exception as e:
        print(f"\nError executing task {task_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to continue...")


def main():
    """Main function to run the program."""
    while True:
        clear_screen()
        display_menu()

        choice = input("\nEnter task number (0-4): ")

        if choice == '0':
            print("Exiting program. Goodbye!")
            sys.exit(0)
        elif choice in ['1', '2', '3', '4']:
            clear_screen()
            print(f"Running Task {choice}...")
            execute_task(choice)
        else:
            print("Invalid choice. Please try again.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        input("\nPress Enter to exit...")