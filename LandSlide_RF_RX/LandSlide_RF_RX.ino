
// ============================================================
// HazardML_RX.ino
// ESP32 RECEIVER: LoRa + ML HAZARD DATA DISPLAY
// ============================================================

#include <Arduino.h>
#include <SPI.h>
#include <LoRa.h>
#include <math.h>

// LoRa pins - must match receiver wiring
const int LORA_SS  = 5;
const int LORA_RST = 14;
const int LORA_DIO0 = 26;

const long LORA_FREQ = 433E6;

unsigned long packetsReceived = 0;

// ------------------------------------------------------------
// HAZARD CLASS NAMES
// ------------------------------------------------------------

const char* getHazardName(int hazardClass) {
  switch (hazardClass) {
    case 0:
      return "LOWER CHANCE OF LANDSLIDE";

    case 1:
      return "LOOKOUT FOR LANDSLIDE";

    case 2:
      return "HIGH RISK FOR LANDSLIDE";

    default:
      return "UNKNOWN CLASS";
  }
}

// ------------------------------------------------------------
// EXTRACT A VALUE FROM THE PACKET
// Example: soil=45.20
// ------------------------------------------------------------

String getValue(const String& packet, const String& key) {
  String searchKey = key + "=";

  int start = packet.indexOf(searchKey);

  if (start < 0) {
    return "";
  }

  start += searchKey.length();

  int end = packet.indexOf(',', start);

  if (end < 0) {
    end = packet.length();
  }

  return packet.substring(start, end);
}

// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println("       LANDSLIDE ML - RECEIVER");
  Serial.println("========================================");

  SPI.begin(18, 19, 23, LORA_SS);

  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

  if (!LoRa.begin(LORA_FREQ)) {
    Serial.println("ERROR: LoRa initialization failed!");
    Serial.println("Check wiring, power, and LoRa module.");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("LoRa initialized at 433 MHz.");
  Serial.println("Waiting for transmitter data...");
}

// ------------------------------------------------------------
// MAIN LOOP
// ------------------------------------------------------------

void loop() {
  int packetSize = LoRa.parsePacket();

  if (packetSize <= 0) {
    return;
  }

  String received;
  received.reserve(packetSize + 1);

  while (LoRa.available()) {
    received += (char)LoRa.read();
  }

  // Accept only the transmitter's ML packet format.
  if (!received.startsWith("ML,")) {
    Serial.println("Ignored non-ML packet.");
    return;
  }

  // Check that the required fields exist.
  String soil  = getValue(received, "soil");
  String rms   = getValue(received, "rms");
  String peak  = getValue(received, "peak");
  String pitch = getValue(received, "pitch");
  String roll  = getValue(received, "roll");
  String cls   = getValue(received, "class");
  String p0    = getValue(received, "p0");
  String p1    = getValue(received, "p1");
  String p2    = getValue(received, "p2");

  if (soil == "" || rms == "" || peak == "" ||
      pitch == "" || roll == "" || cls == "" ||
      p0 == "" || p1 == "" || p2 == "") {
    Serial.println("ERROR: Incomplete ML packet.");
    Serial.println(received);
    return;
  }

  int hazardClass = cls.toInt();

  if (hazardClass < 0 || hazardClass > 2) {
    Serial.println("ERROR: Invalid hazard class.");
    Serial.println(received);
    return;
  }

  packetsReceived++;

  // ----------------------------------------------------------
  // DISPLAY RECEIVED DATA
  // ----------------------------------------------------------

  Serial.println();
  Serial.println("========================================");
  Serial.printf("           PACKET #%lu\n", packetsReceived);
  Serial.println("========================================");

  Serial.println("[SENSOR READINGS]");

  Serial.printf("Soil moisture       : %s %%\n", soil.c_str());
  Serial.printf("Vibration RMS       : %s g\n", rms.c_str());
  Serial.printf("Peak vibration      : %s g\n", peak.c_str());
  Serial.printf("Pitch               : %s degrees\n", pitch.c_str());
  Serial.printf("Roll                : %s degrees\n", roll.c_str());

  Serial.println("----------------------------------------");
  Serial.println("[ML PREDICTION]");

  Serial.printf("Hazard class        : %d\n", hazardClass);
  Serial.printf("Hazard status       : %s\n",
                getHazardName(hazardClass));

  Serial.println("----------------------------------------");
  Serial.println("[CLASS PROBABILITIES]");

  Serial.printf("Class 0 - Lower Chance : %s\n", p0.c_str());
  Serial.printf("Class 1 - Lookout      : %s\n", p1.c_str());
  Serial.printf("Class 2 - High Risk    : %s\n", p2.c_str());

  Serial.println("----------------------------------------");
  Serial.printf("RSSI                : %d dBm\n", LoRa.packetRssi());
  Serial.printf("SNR                 : %.2f dB\n", LoRa.packetSnr());

  Serial.println("----------------------------------------");
  Serial.println("Raw packet:");
  Serial.println(received);

  Serial.println("========================================");
}
