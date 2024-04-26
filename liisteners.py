import subprocess
import time
import os

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_listening_ports():
    """Returns the list of listening ports using the appropriate system command."""
    if os.name == 'nt':  # Windows
        cmd = 'netstat -an | find "LISTENING"'
    else:  # Unix-like (Linux, macOS)
        cmd = "lsof -i -P -n | grep LISTEN"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

while True:
    clear_screen()  # Clear the screen
    print("Listening ports:")
    print(get_listening_ports())  # Print the current listening ports
    time.sleep(2)  # Update interval (2 seconds in this example)
