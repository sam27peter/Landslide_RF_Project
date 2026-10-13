// ============================================================
// LANDSLIDE HAZARD MONITOR
// ESP32 + ADXL345 + SOIL SENSOR + RANDOM FOREST + LoRa
// ============================================================

#include <Arduino.h>
#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <math.h>
#include <stdio.h>
#include "forest_model.h"

using namespace HazardModel;

// ============================================================
// 1. PIN CONFIGURATION — PRESERVED FROM YOUR ORIGINAL CODE
// ============================================================

const int SOIL_PIN = 34;

const int I2C_SDA = 21;
const int I2C_SCL = 22;
const uint8_t ADXL345_ADDRESS = 0x53;

const int LORA_SS = 5;
const int LORA_RST = 14;
const int LORA_DIO0 = 26;
const long LORA_FREQ = 433E6;

// ============================================================
// 2. SOIL SENSOR CALIBRATION
// ============================================================

// Existing ADC calibration values.
// Replace these with measured calibration values if required.

const int SOIL_DRY_RAW = 3000;
const int SOIL_WET_RAW = 1200;

// IMPORTANT:
// This equation produces a calibrated 0-100 percentage.
// It is NOT automatically equivalent to dataset VWC (%).

// ============================================================
// 3. ADXL345 CONFIGURATION
// ============================================================

const float G_PER_COUNT = 0.0039f;

const float SAMPLE_RATE_HZ = 100.0f;
const float GRAVITY_CUTOFF_HZ = 0.3f;
const float WINDOW_SECONDS = 10.0f;

const int WINDOW_SAMPLES =
    (int)(SAMPLE_RATE_HZ * WINDOW_SECONDS + 0.5f);

const int SMOOTHING_SAMPLES = 3;

const float BIAS_X_G = 0.0f;
const float BIAS_Y_G = 0.0f;
const float BIAS_Z_G = 0.0f;

const uint8_t REG_DEVID = 0x00;
const uint8_t REG_BW_RATE = 0x2C;
const uint8_t REG_POWER_CTL = 0x2D;
const uint8_t REG_DATA_FORMAT = 0x31;
const uint8_t REG_DATAX0 = 0x32;

const uint8_t ODR_100_HZ = 0x0A;
const uint8_t FULL_RES_16G = 0x0B;

const float ALPHA = expf(
    -2.0f * PI * GRAVITY_CUTOFF_HZ / SAMPLE_RATE_HZ
);

// ============================================================
// 4. RUNTIME VARIABLES
// ============================================================

bool loraReady = false;

float latestSoilRaw = 0.0f;
float latestSoilPct = 0.0f;

float axG = 0.0f;
float ayG = 0.0f;
float azG = 0.0f;

float gx = 0.0f;
float gy = 0.0f;
float gz = 0.0f;

bool gravityInitialized = false;

float bufferX[SMOOTHING_SAMPLES] = {0};
float bufferY[SMOOTHING_SAMPLES] = {0};
float bufferZ[SMOOTHING_SAMPLES] = {0};

int bufferIndex = 0;
int bufferCount = 0;

float sumX = 0.0f;
float sumY = 0.0f;
float sumZ = 0.0f;

uint32_t windowSampleCount = 0;

double vibrationEnergySum = 0.0;
float peakVibrationMagnitude = 0.0f;

float pitchDeg = 0.0f;
float rollDeg = 0.0f;

double soilSum = 0.0;
uint32_t soilReadCount = 0;

uint32_t nextSampleMicros = 0;
uint32_t nextSoilMillis = 0;
uint32_t packetCounter = 0;

// ============================================================
// 5. I2C REGISTER FUNCTIONS
// ============================================================

bool writeRegister(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(ADXL345_ADDRESS);
    Wire.write(reg);
    Wire.write(value);

    return Wire.endTransmission() == 0;
}

bool readRegisters(
    uint8_t startReg,
    uint8_t *buffer,
    size_t length
) {
    Wire.beginTransmission(ADXL345_ADDRESS);
    Wire.write(startReg);

    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    size_t received = Wire.requestFrom(
        (uint8_t)ADXL345_ADDRESS,
        (uint8_t)length,
        (uint8_t)true
    );

    if (received != length) {
        while (Wire.available()) {
            Wire.read();
        }
        return false;
    }

    for (size_t i = 0; i < length; i++) {
        buffer[i] = (uint8_t)Wire.read();
    }

    return true;
}

// ============================================================
// 6. ADXL345 INITIALIZATION
// ============================================================

bool configureADXL345() {
    uint8_t deviceID = 0;

    if (!readRegisters(REG_DEVID, &deviceID, 1)) {
        Serial.println("ERROR: Cannot read ADXL345.");
        return false;
    }

    if (deviceID != 0xE5) {
        Serial.printf(
            "ERROR: Unexpected ADXL345 ID: 0x%02X\n",
            deviceID
        );
        return false;
    }

    if (!writeRegister(REG_POWER_CTL, 0x00)) {
        return false;
    }

    if (!writeRegister(REG_BW_RATE, ODR_100_HZ)) {
        return false;
    }

    if (!writeRegister(REG_DATA_FORMAT, FULL_RES_16G)) {
        return false;
    }

    if (!writeRegister(REG_POWER_CTL, 0x08)) {
        return false;
    }

    delay(20);

    Serial.println("ADXL345 detected.");
    Serial.println("FULL_RES enabled.");
    Serial.println("Output data rate: 100 Hz.");

    return true;
}

// ============================================================
// 7. READ RAW ACCELERATION
// ============================================================

bool readRawCounts(
    int16_t &rawX,
    int16_t &rawY,
    int16_t &rawZ
) {
    uint8_t data[6];

    if (!readRegisters(REG_DATAX0, data, 6)) {
        return false;
    }

    rawX = (int16_t)(
        ((uint16_t)data[1] << 8) | data[0]
    );

    rawY = (int16_t)(
        ((uint16_t)data[3] << 8) | data[2]
    );

    rawZ = (int16_t)(
        ((uint16_t)data[5] << 8) | data[4]
    );

    return true;
}

// ============================================================
// 8. SOIL SENSOR
// ============================================================

float readSoilRaw() {
    uint32_t total = 0;

    for (int i = 0; i < 10; i++) {
        total += analogRead(SOIL_PIN);
        delay(1);
    }

    return total / 10.0f;
}

float convertSoilToPercent(float raw) {
    float denominator =
        (float)(SOIL_DRY_RAW - SOIL_WET_RAW);

    if (fabsf(denominator) < 1.0f) {
        return 0.0f;
    }

    float percentage =
        ((SOIL_DRY_RAW - raw) / denominator) * 100.0f;

    return constrain(percentage, 0.0f, 100.0f);
}

// ============================================================
// 9. THREE-SAMPLE MOVING AVERAGE
// ============================================================

void smoothAcceleration(
    float x,
    float y,
    float z,
    float &outX,
    float &outY,
    float &outZ
) {
    if (bufferCount == SMOOTHING_SAMPLES) {
        sumX -= bufferX[bufferIndex];
        sumY -= bufferY[bufferIndex];
        sumZ -= bufferZ[bufferIndex];
    } else {
        bufferCount++;
    }

    bufferX[bufferIndex] = x;
    bufferY[bufferIndex] = y;
    bufferZ[bufferIndex] = z;

    sumX += x;
    sumY += y;
    sumZ += z;

    bufferIndex =
        (bufferIndex + 1) % SMOOTHING_SAMPLES;

    outX = sumX / bufferCount;
    outY = sumY / bufferCount;
    outZ = sumZ / bufferCount;
}

// ============================================================
// 10. SENSOR PROCESSING PIPELINE
// ============================================================

bool sampleADXL345() {
    int16_t rawX, rawY, rawZ;

    if (!readRawCounts(rawX, rawY, rawZ)) {
        Serial.println("WARNING: ADXL345 read failed.");
        return false;
    }

    // Step 1: Convert raw counts to g and apply bias.

    float calibratedX = rawX * G_PER_COUNT - BIAS_X_G;
    float calibratedY = rawY * G_PER_COUNT - BIAS_Y_G;
    float calibratedZ = rawZ * G_PER_COUNT - BIAS_Z_G;

    // Step 2: Smooth the acceleration.

    smoothAcceleration(
        calibratedX,
        calibratedY,
        calibratedZ,
        axG,
        ayG,
        azG
    );

    // Step 3: Estimate gravity using EMA.

    if (!gravityInitialized) {
        gx = axG;
        gy = ayG;
        gz = azG;

        gravityInitialized = true;
    } else {
        gx = ALPHA * gx + (1.0f - ALPHA) * axG;
        gy = ALPHA * gy + (1.0f - ALPHA) * ayG;
        gz = ALPHA * gz + (1.0f - ALPHA) * azG;
    }

    // Step 4: Remove estimated gravity.

    float vx = axG - gx;
    float vy = ayG - gy;
    float vz = azG - gz;

    float magnitudeSquared =
        vx * vx + vy * vy + vz * vz;

    float magnitude = sqrtf(magnitudeSquared);

    vibrationEnergySum += magnitudeSquared;

    if (magnitude > peakVibrationMagnitude) {
        peakVibrationMagnitude = magnitude;
    }

    windowSampleCount++;

    // Step 5: Calculate tilt from filtered gravity.

    pitchDeg = atan2f(
        -gx,
        sqrtf(gy * gy + gz * gz)
    ) * 180.0f / PI;

    rollDeg = atan2f(
        gy,
        gz
    ) * 180.0f / PI;

    return true;
}

// ============================================================
// 11. RANDOM FOREST INFERENCE
// Uses ALL exported trees, not only the first 10.
// ============================================================

bool predictTree(
    const Node *tree,
    int nodeCount,
    const float features[NUM_FEATURES],
    float probabilities[NUM_CLASSES]
) {
    int nodeIndex = 0;

    while (true) {
        if (nodeIndex < 0 || nodeIndex >= nodeCount) {
            return false;
        }

        const Node &node = tree[nodeIndex];

        // A leaf has no children.

        if (node.left < 0 && node.right < 0) {
            probabilities[0] = node.p0;
            probabilities[1] = node.p1;
            probabilities[2] = node.p2;

            return true;
        }

        int featureIndex = node.feature;

        if (
            featureIndex < 0 ||
            featureIndex >= NUM_FEATURES
        ) {
            return false;
        }

        if (features[featureIndex] <= node.threshold) {
            nodeIndex = node.left;
        } else {
            nodeIndex = node.right;
        }
    }
}

bool predictHazard(
    const float features[NUM_FEATURES],
    float probabilities[NUM_CLASSES]
) {
    for (int c = 0; c < NUM_CLASSES; c++) {
        probabilities[c] = 0.0f;
    }

    float treeProbabilities[NUM_CLASSES];

    for (int t = 0; t < NUM_TREES; t++) {
        if (!predictTree(
            FOREST_TREES[t],
            FOREST_NODE_COUNTS[t],
            features,
            treeProbabilities
        )) {
            return false;
        }

        for (int c = 0; c < NUM_CLASSES; c++) {
            probabilities[c] += treeProbabilities[c];
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        probabilities[c] /= NUM_TREES;
    }

    return true;
}

int getPredictedClass(
    const float probabilities[NUM_CLASSES]
) {
    int bestClass = 0;

    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probabilities[c] > probabilities[bestClass]) {
            bestClass = c;
        }
    }

    return bestClass;
}

const char *getHazardName(int classID) {
    switch (classID) {
        case 0:
            return "Lower Chance of Landslide";

        case 1:
            return "Lookout for Landslide";

        case 2:
            return "High Risk for Landslide";

        default:
            return "UNKNOWN";
    }
}

// ============================================================
// 12. TEN-SECOND WINDOW PROCESSING
// ============================================================

void processWindow() {
    if (windowSampleCount == 0) {
        Serial.println("No valid ADXL345 samples.");
        return;
    }

    float vibrationRMS = sqrtf(
        vibrationEnergySum / windowSampleCount
    );

    float soilPct = soilReadCount > 0
        ? soilSum / soilReadCount
        : latestSoilPct;

    // MUST MATCH THE PYTHON MODEL'S FEATURE ORDER:
    // 0 soil VWC (%)
    // 1 vibration RMS (g)
    // 2 peak vibration magnitude (g)
    // 3 pitch (degrees)
    // 4 roll (degrees)

    float features[NUM_FEATURES] = {
        soilPct,
        vibrationRMS,
        peakVibrationMagnitude,
        pitchDeg,
        rollDeg
    };

    float probabilities[NUM_CLASSES] = {0.0f};

    bool predictionOK = predictHazard(
        features,
        probabilities
    );

    Serial.println();
    Serial.println("========== 10-SECOND ML RESULT ==========");

    Serial.printf(
        "ADXL samples: %lu\n",
        (unsigned long)windowSampleCount
    );

    Serial.printf("Soil ADC: %.0f\n", latestSoilRaw);
    Serial.printf("Soil percentage: %.2f %%\n", soilPct);
    Serial.printf("Vibration RMS: %.5f g\n", vibrationRMS);

    Serial.printf(
        "Peak vibration: %.5f g\n",
        peakVibrationMagnitude
    );

    Serial.printf("Pitch: %.2f degrees\n", pitchDeg);
    Serial.printf("Roll: %.2f degrees\n", rollDeg);

    if (!predictionOK) {
        Serial.println("ERROR: Random Forest traversal failed.");
        Serial.println("Prediction not transmitted.");
    } else {
        int predictedClass = getPredictedClass(probabilities);

        Serial.printf(
            "Class 0 probability: %.6f\n",
            probabilities[0]
        );

        Serial.printf(
            "Class 1 probability: %.6f\n",
            probabilities[1]
        );

        Serial.printf(
            "Class 2 probability: %.6f\n",
            probabilities[2]
        );

        Serial.printf(
            "PREDICTED CLASS: %d\n",
            predictedClass
        );

        Serial.printf(
            "HAZARD CATEGORY: %s\n",
            getHazardName(predictedClass)
        );

        // Preserve the existing receiver packet field names.

        if (loraReady) {
            char packet[240];

            snprintf(
                packet,
                sizeof(packet),
                "ML,soil=%.2f,rms=%.5f,peak=%.5f,"
                "pitch=%.2f,roll=%.2f,class=%d,"
                "p0=%.3f,p1=%.3f,p2=%.3f",
                soilPct,
                vibrationRMS,
                peakVibrationMagnitude,
                pitchDeg,
                rollDeg,
                predictedClass,
                probabilities[0],
                probabilities[1],
                probabilities[2]
            );

            LoRa.beginPacket();
            LoRa.print(packet);

            int result = LoRa.endPacket();

            if (result == 1) {
                packetCounter++;

                Serial.printf(
                    "LoRa packet %lu sent:\n",
                    (unsigned long)packetCounter
                );

                Serial.println(packet);
            } else {
                Serial.println("LoRa transmission failed.");
            }
        } else {
            Serial.println("LoRa unavailable.");
        }
    }

    Serial.println("=========================================");

    // Reset the current window.
    // Keep the gravity EMA and smoothing state running.

    windowSampleCount = 0;
    vibrationEnergySum = 0.0;
    peakVibrationMagnitude = 0.0f;

    soilSum = 0.0;
    soilReadCount = 0;
}

// ============================================================
// 13. SETUP
// ============================================================

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("ESP32 LANDSLIDE ML TRANSMITTER");

    Serial.printf("Model trees: %d\n", NUM_TREES);
    Serial.printf("Model features: %d\n", NUM_FEATURES);
    Serial.printf("Model classes: %d\n", NUM_CLASSES);

    pinMode(SOIL_PIN, INPUT);

    analogReadResolution(12);
    analogSetPinAttenuation(SOIL_PIN, ADC_11db);

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);

    if (!configureADXL345()) {
        Serial.println("Check ADXL345 wiring and address.");

        while (true) {
            delay(1000);
        }
    }

    Serial.printf("EMA alpha: %.6f\n", ALPHA);
    Serial.printf("Sampling rate: %.0f Hz\n", SAMPLE_RATE_HZ);
    Serial.printf("Window: %.0f seconds\n", WINDOW_SECONDS);

    Serial.printf(
        "Bias X/Y/Z: %.4f, %.4f, %.4f g\n",
        BIAS_X_G,
        BIAS_Y_G,
        BIAS_Z_G
    );

    LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

    if (LoRa.begin(LORA_FREQ)) {
        loraReady = true;
        Serial.println("LoRa initialized at 433 MHz.");
    } else {
        loraReady = false;
        Serial.println("LoRa initialization failed.");
    }

    Serial.println("Sensor readings: every second.");
    Serial.println("ML predictions: every 10 seconds.");

    Serial.println(
        "WARNING: Predictions are experimental until "
        "sensor feature units are calibrated."
    );

    nextSampleMicros = micros();
    nextSoilMillis = millis() + 1000;
}

// ============================================================
// 14. MAIN LOOP
// ============================================================

void loop() {
    // ADXL345 sampling at approximately 100 Hz.

    uint32_t nowMicros = micros();

    if ((int32_t)(nowMicros - nextSampleMicros) >= 0) {
        nextSampleMicros +=
            (uint32_t)(1000000.0f / SAMPLE_RATE_HZ);

        sampleADXL345();
    }

    uint32_t nowMillis = millis();

    // Soil reading and serial output once per second.

    if ((int32_t)(nowMillis - nextSoilMillis) >= 0) {
        nextSoilMillis += 1000;

        latestSoilRaw = readSoilRaw();
        latestSoilPct = convertSoilToPercent(latestSoilRaw);

        soilSum += latestSoilPct;
        soilReadCount++;

        Serial.printf(
            "Soil ADC: %.0f | Moisture: %.2f %% | "
            "Accel(g): X=%.3f Y=%.3f Z=%.3f | "
            "Gravity(g): X=%.3f Y=%.3f Z=%.3f\n",
            latestSoilRaw,
            latestSoilPct,
            axG,
            ayG,
            azG,
            gx,
            gy,
            gz
        );

        // Process a window after approximately ten seconds.

        if (soilReadCount >= (uint32_t)WINDOW_SECONDS) {
            processWindow();
        }
    }

    delayMicroseconds(100);
}