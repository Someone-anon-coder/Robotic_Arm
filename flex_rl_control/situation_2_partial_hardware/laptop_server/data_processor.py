# File: flex_rl_control/situation_2_partial_hardware/laptop_server/data_processor.py

import socket
import json
import threading

class ESP32DataProcessor:
    """
    Runs a UDP Server on a background thread.
    Listens for incoming JSON packets from the ESP32 Glove and parses them.
    """
    def __init__(self, ip="0.0.0.0", port=5005):
        self.ip = ip
        self.port = port
        
        # Default safe values before the ESP32 connects
        self.latest_data = {
            "flex_thumb": 0,
            "flex_index": 0,
            "flex_middle": 0,
            "flex_ring": 0,
            "flex_pinky": 0,
            "imu_euler": [0.0, 0.0, 0.0]
        }
        
        # Setup UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0) # 1 second timeout to allow clean thread exits
        
        self.running = True
        
        # Start background listening thread
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[NETWORK] UDP Server listening for ESP32 on port {self.port}...")

    def _receive_loop(self):
        while self.running:
            try:
                # Receive packet
                data, addr = self.sock.recvfrom(1024)
                payload = data.decode('utf-8')
                
                # Parse JSON
                parsed_data = json.loads(payload)
                
                # Safely update the dictionary PyBullet will read from
                self.latest_data = parsed_data
                
            except socket.timeout:
                pass # Timeout is expected, just loop again
            except json.JSONDecodeError:
                print("[WARNING] Received malformed JSON packet from ESP32.")
            except Exception as e:
                print(f"[ERROR] UDP Receiver Exception: {e}")

    def get_sensor_data(self):
        """ Returns the most recent parsed data dict from the ESP32. """
        return self.latest_data

    def close(self):
        """ Cleanly stops the background thread and closes the socket. """
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.sock.close()
        print("[NETWORK] UDP Server shut down securely.")