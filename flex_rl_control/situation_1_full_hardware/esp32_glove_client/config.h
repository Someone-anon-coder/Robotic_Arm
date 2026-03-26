// File: flex_rl_control/situation_1_full_hardware/esp32_glove_client/config.h

#ifndef CONFIG_H
#define CONFIG_H

// --- Wi-Fi Setup ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// --- Laptop Server Setup ---
// The IP address of your laptop on the local network
const char* server_ip = "YOUR_LAPTOP_IP"; 
const int server_port = 5005;

// --- Flex Sensor Pin Definitions (ADC Pins) ---
const int PIN_FLEX_THUMB  = 32;
const int PIN_FLEX_INDEX  = 33;
const int PIN_FLEX_MIDDLE = 34;
const int PIN_FLEX_RING   = 35;
const int PIN_FLEX_PINKY  = 36;

// I2C Pins for MPU6050 (Default ESP32)
// SDA = 21, SCL = 22

#endif