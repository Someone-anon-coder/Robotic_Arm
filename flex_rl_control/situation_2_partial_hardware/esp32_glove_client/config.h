// File: flex_rl_control/situation_2_partial_hardware/esp32_glove_client/config.h

#ifndef CONFIG_H
#define CONFIG_H

// --- Network Configuration ---
// Enter your local Wi-Fi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Enter the IPv4 address of the LAPTOP running the Python simulation
const char* server_ip = "YOUR_LAPTOP_IP"; // <-- CHANGE THIS
const int server_port = 5005;            // UDP Port

// --- Hardware Pins ---
// ESP32 ADC Pins for the 5 Flex Sensors
const int PIN_FLEX_THUMB  = 32;
const int PIN_FLEX_INDEX  = 33;
const int PIN_FLEX_MIDDLE = 34;
const int PIN_FLEX_RING   = 35;
const int PIN_FLEX_PINKY  = 36;

// MPU6050 I2C Pins (Standard ESP32 I2C)
// SDA = 21
// SCL = 22

#endif