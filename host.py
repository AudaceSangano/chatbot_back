import socket

def get_hostname(ip):
    try:
        hostname = socket.gethostbyaddr(ip)
        return hostname[0]
    except socket.herror:
        return "Unknown host"

# Example usage:
ip_address = "192.168.1.120"  # Replace with any IP in your network
hostname = get_hostname(ip_address)
print(f"IP: {ip_address} | Hostname: {hostname}")
