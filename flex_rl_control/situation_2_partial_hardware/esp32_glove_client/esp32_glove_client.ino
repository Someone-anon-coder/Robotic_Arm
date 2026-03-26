// File: flex_rl_control/situation_2_partial_hardware/esp32_glove_client/esp32_glove_client.ino

#include <WiFi.h>
#include <WiFiUDP.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h> // Install via Library Manager
#include "config.h"

WiFiUDP udp;
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);

  // 1. Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // 2. Initialize MPU6050 IMU
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip. Check wiring!");
  } else {
    Serial.println("MPU6050 Found!");
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // 3. Configure ADC (ESP32 uses 12-bit ADC: 0 to 4095)
  analogReadResolution(12); 
}

void loop() {
  // A. Read Flex Sensors & Map to 0-1023 (To match our simulation format)
  int f_thumb = map(analogRead(PIN_FLEX_THUMB), 0, 4095, 0, 1023);
  int f_index = map(analogRead(PIN_FLEX_INDEX), 0, 4095, 0, 1023);
  int f_mid   = map(analogRead(PIN_FLEX_MIDDLE), 0, 4095, 0, 1023);
  int f_ring  = map(analogRead(PIN_FLEX_RING), 0, 4095, 0, 1023);
  int f_pinky = map(analogRead(PIN_FLEX_PINKY), 0, 4095, 0, 1023);

  // B. Read IMU
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Calculate simple Roll and Pitch in radians from Accelerometer data
  float roll = atan2(a.acceleration.y, a.acceleration.z);
  float pitch = atan2(-a.acceleration.x, sqrt(a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z));
  float yaw = 0.0; // Yaw drifts without a magnetometer, keeping 0.0 for now

  // C. Create JSON Payload
  JsonDocument doc; // ArduinoJson v7 syntax
  doc["flex_thumb"]  = f_thumb;
  doc["flex_index"]  = f_index;
  doc["flex_middle"] = f_mid;
  doc["flex_ring"]   = f_ring;
  doc["flex_pinky"]  = f_pinky;

  JsonArray imu = doc["imu_euler"].to<JsonArray>();
  imu.add(roll);
  imu.add(pitch);
  imu.add(yaw);

  String payload;
  serializeJson(doc, payload);

  // D. Send via UDP
  udp.beginPacket(server_ip, server_port);
  udp.print(payload);
  udp.endPacket();

  // Print to Serial for debugging
  Serial.println(payload);

  // Run at roughly ~50Hz
  delay(20); 
}