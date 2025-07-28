from scapy.all import ARP, Ether, srp
import socket

# Function to discover devices on the network
def scan_network(ip_range):
    print(f"[+] Scanning network range: {ip_range}")
    arp = ARP(pdst=ip_range)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether/arp

    result = srp(packet, timeout=2, verbose=0)[0]
    devices = []
    for sent, received in result:
        devices.append({'ip': received.psrc, 'mac': received.hwsrc})
    return devices

# Function to scan ports on a given IP address
def scan_ports(ip, start_port=1, end_port=100):
    print(f"    [*] Scanning ports on {ip}")
    open_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            result = s.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
    return open_ports

# Main logic
if __name__ == "__main__":
    target_range = "192.168.1.0/24"  # Adjust to match your local network
    devices = scan_network(target_range)

    if not devices:
        print("[-] No devices found on the network.")
    else:
        print(f"[+] Found {len(devices)} device(s):\n")
        for device in devices:
            print(f"IP: {device['ip']}, MAC: {device['mac']}")
            open_ports = scan_ports(device['ip'])
            if open_ports:
                print(f"    [+] Open ports: {open_ports}\n")
            else:
                print("    [-] No open ports found.\n")

