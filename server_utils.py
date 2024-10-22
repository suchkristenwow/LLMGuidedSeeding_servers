import psutil
import time

# Define the ports where the servers are running
server_ports = {
    "yolo_world_server_with_seg": 5005,
    "qwen_server": 5007,
    "segment_anything_server": 5006
}

# Function to get the process ID (PID) from a port number
def get_pid_from_port(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            return conn.pid
    return None

# Function to monitor memory usage for a given process
def monitor_memory_usage(pid, server_name):
    try:
        process = psutil.Process(pid)
        while True:
            mem_info = process.memory_info()
            rss = mem_info.rss / (1024 * 1024)  # Convert to MB
            vms = mem_info.vms / (1024 * 1024)  # Convert to MB
            print(f"[{server_name}] PID: {pid} | RSS: {rss:.2f} MB | VMS: {vms:.2f} MB")
            time.sleep(2)  # Adjust the interval to control how often memory usage is printed
    except psutil.NoSuchProcess:
        print(f"[{server_name}] Process with PID {pid} not found. It might have exited.")

# Main function to monitor all server processes
def monitor_servers_memory():
    for server_name, port in server_ports.items():
        pid = get_pid_from_port(port)
        if pid:
            print(f"Monitoring memory usage for {server_name} (PID: {pid}, Port: {port})")
            monitor_memory_usage(pid, server_name)
        else:
            print(f"Could not find a process running on port {port} for {server_name}.")

if __name__ == "__main__":
    monitor_servers_memory()
