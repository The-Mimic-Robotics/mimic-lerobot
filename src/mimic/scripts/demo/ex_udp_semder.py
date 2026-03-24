import socket
import sys

# --- UDP Configuration ---
# If your solver and robot script are on the SAME computer, use "127.0.0.1"
# If they are on DIFFERENT computers, change this to the robot computer's IP address (e.g., "192.168.1.50")
UDP_IP = "127.0.0.1" 
UDP_PORT = 5005

def main():
    print(f"--- UDP Mock Solver Sender ---")
    print(f"Targeting IP: {UDP_IP} | Port: {UDP_PORT}")
    print("\nAvailable Commands:")
    print("  [0-8]  -> Replay Tic-Tac-Toe placement episode")
    print("  [-1]   -> Switch to Teleoperation")
    print("  [w]    -> Wait / Pause robot")
    print("  [exit] -> Quit this sender script")
    print("-" * 30)

    # Set up the UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while True:
            # Get input from the terminal
            command = input("\nEnter command to send: ").strip()
            
            # Handle local exit command
            if command.lower() == 'exit':
                print("Shutting down sender...")
                break
                
            # Prevent sending empty strings
            if not command:
                continue

            # Send the command via UDP
            sock.sendto(command.encode('utf-8'), (UDP_IP, UDP_PORT))
            print(f" [Sent] -> '{command}'")

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt. Shutting down sender...")
    finally:
        sock.close()
        sys.exit(0)

if __name__ == "__main__":
    main()