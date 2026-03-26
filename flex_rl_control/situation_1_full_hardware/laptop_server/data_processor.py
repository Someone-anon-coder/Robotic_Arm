# File: flex_rl_control/situation_1_full_hardware/laptop_server/data_processor.py

import socket
import json
import threading

class ESP32DataProcessor:
    """
    UDP Server running on a background thread.
    Receives JSON packets from the ESP32 Glove and updates local storage.
    """
    def __init__(self, ip="0.0.0.0", port=5005):
        self.ip = ip
        self.port = port
        self.latest_data = {
            "flex_thumb": 0, "flex_index": 0, "flex_middle": 0,
            "flex_ring": 0, "flex_pinky": 0,
            "imu_euler": [0.0, 0.0, 0.0]
        }
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0)
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[NETWORK] Laptop Receiver listening for ESP32 on port {self.port}")

    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                self.latest_data = json.loads(data.decode('utf-8'))
            except (socket.timeout, json.JSONDecodeError, Exception):
                continue

    def get_sensor_data(self):
        return self.latest_data

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.sock.close()