// File: flex_rl_control/situation_1_full_hardware/esp32_glove_client/esp32_glove_client.ino

#include <WiFi.h>
#include <WiFiUDP.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>
#include "config.h"

WiFiUDP udp;
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected.");

  // Initialize IMU
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("MPU6050 Ready.");
  }

  analogReadResolution(12); // ESP32 12-bit resolution
}

void loop() {
  // Read Flex Sensors and map 0-4095 to 0-1023
  int f0 = map(analogRead(PIN_FLEX_THUMB), 0, 4095, 0, 1023);
  int f1 = map(analogRead(PIN_FLEX_INDEX), 0, 4095, 0, 1023);
  int f2 = map(analogRead(PIN_FLEX_MIDDLE), 0, 4095, 0, 1023);
  int f3 = map(analogRead(PIN_FLEX_RING), 0, 4095, 0, 1023);
  int f4 = map(analogRead(PIN_FLEX_PINKY), 0, 4095, 0, 1023);

  // Read IMU Roll/Pitch
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float roll  = atan2(a.acceleration.y, a.acceleration.z);
  float pitch = atan2(-a.acceleration.x, sqrt(a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z));

  // Build JSON
  JsonDocument doc;
  doc["flex_thumb"]  = f0;
  doc["flex_index"]  = f1;
  doc["flex_middle"] = f2;
  doc["flex_ring"]   = f3;
  doc["flex_pinky"]  = f4;
  
  JsonArray imu = doc["imu_euler"].to<JsonArray>();
  imu.add(roll);
  imu.add(pitch);
  imu.add(0.0); // Yaw

  String output;
  serializeJson(doc, output);

  // Send UDP packet to Laptop
  udp.beginPacket(server_ip, server_port);
  udp.print(output);
  udp.endPacket();

  Serial.println(output);
  delay(20); // ~50Hz
}