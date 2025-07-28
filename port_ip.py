import socket
from datetime import datetime

# Ask user for IP to scan
target = input("Enter IP address to scan: ")

# Display starting time
print(f"\nScanning target: {target}")
print(f"Time started: {datetime.now()}")
print("-" * 50)

# Define port range
start_port = 1
end_port = 1024

# Start scanning
try:
    for port in range(start_port, end_port + 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        result = s.connect_ex((target, port))
        if result == 0:
            print(f"Port {port} is OPEN")
        s.close()

except KeyboardInterrupt:
    print("\nScan cancelled by user.")
except socket.gaierror:
    print("Hostname could not be resolved.")
except socket.error:
    print("Could not connect to server.")

# Display ending time
print("-" * 50)
print(f"Time finished: {datetime.now()}")
