import os
import socket
import time
from DashboardGUI import DashboardGUI


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(start_port: int) -> int:
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port


if __name__ == "__main__":
    base_port = int(os.environ.get("DASHBOARD_PORT", "8050"))
    port = find_free_port(base_port)
    if port != base_port:
        print(f"Port {base_port} is in use, using {port} instead.")

    hub = DashboardGUI(agent=None, base_log_dir="logs")
    hub.launch_dashboard_gui(port=port)
    print(f"Dashboard hub running at http://localhost:{port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        hub.stop()
