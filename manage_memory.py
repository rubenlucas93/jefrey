import sys
from memory.storage import Memory

def show_help():
    print("Usage: python manage_memory.py [COMMAND]")
    print("\nAvailable Commands:")
    print("  clear       : Completely wipes all persisted memories.")
    print("  expire [N]  : Deletes all memories older than [N] days (e.g., expire 30)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)

    command = sys.argv[1].lower()
    mem = Memory()

    if command == "clear":
        print("WARNING: This will delete ALL past conversations permanently.")
        confirm = input("Are you sure? (type 'yes' to proceed): ")
        if confirm.lower() == "yes":
            mem.clear_all()
        else:
            print("Action canceled.")
            
    elif command == "expire":
        if len(sys.argv) < 3:
            print("Error: You must provide the number of days.")
            print("Example: python manage_memory.py expire 30")
            sys.exit(1)
            
        try:
            days = int(sys.argv[2])
            mem.expire_older_than(days)
        except ValueError:
            print("Error: Days must be a valid number.")
            
    else:
        print(f"Unknown command: {command}")
        show_help()
