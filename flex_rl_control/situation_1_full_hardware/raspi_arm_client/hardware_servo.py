# File: flex_rl_control/situation_1_full_hardware/raspi_arm_client/hardware_servo.py

import math

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[WARNING] Adafruit libraries not found. Running in MOCK hardware mode.")
    print("To install on Raspi: pip install adafruit-circuitpython-pca9685 adafruit-circuitpython-servokit")

class ArmHardwareController:
    """
    Interfaces with the PCA9685 Servo Driver via I2C to control the MG995 servos.
    """
    def __init__(self):
        self.servos = {}
        if HARDWARE_AVAILABLE:
            # Initialize I2C bus and PCA9685 board
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c)
            self.pca.frequency = 50 # Standard 50Hz for MG995 servos
            
            # Map the specific PCA9685 channels to your physical build!
            # Change the indices below depending on where you plug the servos in.
            self.servos = {
                "thumb_angle": servo.Servo(self.pca.channels[0]),
                "index_angle": servo.Servo(self.pca.channels[1]),
                "middle_angle": servo.Servo(self.pca.channels[2]),
                "ring_angle": servo.Servo(self.pca.channels[3]),
                "pinky_angle": servo.Servo(self.pca.channels[4]),
                "wrist_angle": servo.Servo(self.pca.channels[5])
            }
        print("[HARDWARE] Servo Controller Initialized.")

    def update_servos(self, motor_commands):
        """
        Takes the dictionary of motor commands (in radians) from the RL agent 
        and applies them physically to the servos.
        """
        for joint, rad_value in motor_commands.items():
            if joint in self.servos:
                # Convert radians from RL model to Degrees for the physical servo
                deg_value = math.degrees(rad_value)
                
                # IMPORTANT TENSION CALIBRATION:
                # You may need to multiply deg_value by a factor (e.g., * 1.5) if 
                # 90 degrees of servo rotation isn't pulling the tendon far enough.
                
                # Clamp safely between 0 and 180 to protect the servos
                deg_value = max(0.0, min(180.0, deg_value))

                if HARDWARE_AVAILABLE:
                    # Physically actuate the motor
                    self.servos[joint].angle = deg_value
                else:
                    # Mock mode (silent)
                    pass