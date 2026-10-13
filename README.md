# Landslide Hazard Prediction Using ESP32 and Random Forest

An embedded machine-learning project that uses soil moisture and ground-movement measurements to classify landslide hazard conditions. A Random Forest model runs directly on an ESP32, and LoRa communication sends the prediction to a second ESP32 for monitoring.

## Project Overview

The system collects sensor readings, calculates features, performs on-device Random Forest inference, and transmits the result over a 433 MHz LoRa link.

### Main Features

- Soil moisture measurement using an HW-390 capacitive soil moisture sensor.
- Acceleration, vibration, and tilt measurements using an ADXL345 accelerometer.
- Random Forest classification implemented on the ESP32.
- Model conversion from Python to a C++ header for embedded inference.
- LoRa communication between transmitter and receiver ESP32 boards.
- Serial Monitor output for sensor readings, predicted class, class probabilities, RSSI, and SNR.

## System Architecture

```text
HW-390 Soil Moisture Sensor
             |
             v
       Sender ESP32 <--- ADXL345
             |
             v
    Feature Calculation
             |
             v
  Random Forest: 100 Trees
             |
             v
     Hazard Classification
             |
             v
       RA-02 LoRa TX
             |
       433 MHz Link
             |
             v
       RA-02 LoRa RX
             |
             v
     Receiver ESP32
             |
             v
       Serial Monitor
```

The sender prints sensor readings approximately every second and sends a prediction packet approximately every 10 seconds, according to the configured firmware timing.

## Hardware Requirements

- ESP32 development boards (two)
- HW-390 capacitive soil moisture sensor
- ADXL345 accelerometer
- RA-02 LoRa modules based on the SX1278 radio (two)
- Jumper wires and breadboard
- Suitable antenna for the LoRa frequency
- Stable power supply appropriate for each module

## Pin Connections

### HW-390 Soil Moisture Sensor — Sender ESP32

| Sensor pin | ESP32   |
| ---------- | ------- |
| VCC        | 3V3     |
| GND        | GND     |
| AOUT       | GPIO 34 |

### ADXL345 — I²C Mode

| ADXL345 pin | ESP32              |
| ----------- | ------------------ |
| VCC         | 3V3                |
| GND         | GND                |
| CS          | 3V3                |
| SDA         | GPIO 21            |
| SCL         | GPIO 22            |
| SDO         | GND                |
| INT1, INT2  | Leave disconnected |

### RA-02 LoRa Module — Both ESP32 Boards

| RA-02 pin | ESP32              |
| --------- | ------------------ |
| VCC       | Regulated 3.3 V    |
| GND       | GND                |
| SCK       | GPIO 18            |
| MISO      | GPIO 19            |
| MOSI      | GPIO 23            |
| NSS / CS  | GPIO 5             |
| RESET     | GPIO 14            |
| DIO0      | GPIO 26            |
| DIO1      | GPIO 27 (optional) |

The sender and receiver must use compatible LoRa settings and the same configured frequency. Never power the RA-02 from 5 V. Connect a suitable antenna before transmitting.

## Machine Learning Model

The project uses a Random Forest classifier trained in Python and exported for execution on the ESP32.

The embedded model contains 100 decision trees and uses five features in the following exact order:

| Index | Feature                 |
| ----: | ----------------------- |
|     0 | `soil_moisture_vwc_pct` |
|     1 | `accel_vib_rms_g`       |
|     2 | `accel_peak_mag_g`      |
|     3 | `tilt_pitch_deg`        |
|     4 | `tilt_roll_deg`         |

The firmware evaluates the trees, averages their class probabilities, and selects the class with the highest probability.

### Output Classes

| Class | Meaning                   |
| ----: | ------------------------- |
|     0 | Lower Chance of Landslide |
|     1 | Lookout for Landslide     |
|     2 | High Risk for Landslide   |

These labels are the model's classifications, not guarantees about future events.

## LoRa Data Format

The sender transmits a text packet in the following format:

```text
ML,soil=92.00,rms=0.00420,peak=0.02100,pitch=2.50,roll=-1.20,class=1,p0=0.150,p1=0.700,p2=0.150
```

The values above are an illustrative example, not actual sensor readings.

| Field            | Description                                 |
| ---------------- | ------------------------------------------- |
| `soil`           | Soil moisture feature                       |
| `rms`            | Vibration RMS                               |
| `peak`           | Peak vibration magnitude                    |
| `pitch`          | Pitch angle                                 |
| `roll`           | Roll angle                                  |
| `class`          | Predicted class                             |
| `p0`, `p1`, `p2` | Model probabilities for classes 0, 1, and 2 |

The receiver also reports LoRa signal strength (RSSI) and signal-to-noise ratio (SNR).

## Software Requirements

- Python 3.11 or a compatible Python version
- NumPy
- pandas
- scikit-learn
- joblib
- Matplotlib
- Arduino IDE
- ESP32 board support package for Arduino IDE
- LoRa library compatible with the SX1278 module

Install the Python packages using:

```bash
pip install numpy pandas scikit-learn joblib matplotlib
```

## Project Structure

```text
Landslide_RF_Project/
├── data/
│   └── idukki_landslide_dataset.csv
├── model/
│   ├── random_forest.joblib
│   └── forest_model.h
├── results/
├── src/
│   ├── train_model.py
│   └── export_to_cpp.py
├── Landslide_RF_TX/
│   ├── Landslide_RF_TX.ino
│   └── forest_model.h
├── Landslide_RF_RX/
│   └── Landslide_RF_RX.ino
├── .gitignore
├── requirements.txt
└── README.md
```

_The directory names above describe the intended organization. Adjust them to match the files actually present in your local repository._

## Running the Project

### 1. Train the model

From the project root, run:

```bash
python src/train_model.py
```

This uses the configured dataset and saves the trained model to the `model` directory.

### 2. Export the model to C++

Run:

```bash
python src/export_to_cpp.py
```

The exporter generates `model/forest_model.h`. Copy the regenerated header into the Arduino transmitter sketch folder so the firmware compiles against the current model.

### 3. Upload the transmitter

1. Open `Landslide_RF_TX.ino` in Arduino IDE.
2. Confirm that `forest_model.h` is in the same sketch folder.
3. Select the appropriate ESP32 board and port.
4. Compile and upload the sketch.

### 4. Upload the receiver

1. Open `Landslide_RF_RX.ino` in Arduino IDE.
2. Select the receiver ESP32 board and port.
3. Compile and upload the sketch.
4. Open Serial Monitor at 115200 baud.

### 5. Test communication

Power both boards and verify that the receiver displays incoming packets, sensor features, the predicted class, probabilities, RSSI, and SNR.

## Validation and Limitations

- Verify embedded predictions against Python predictions using identical feature vectors.
- Calibrate the HW-390 sensor against an appropriate soil-moisture reference. An ADC-derived percentage is not automatically equivalent to volumetric water content (VWC).
- Confirm that vibration RMS, peak magnitude, pitch, and roll are calculated consistently with the training data.
- Evaluate the model using independent data representing different locations, soil conditions, and weather conditions.
- Check performance for each class, including false alarms and missed high-risk events.
- Treat model probabilities as classifier outputs, not automatically calibrated real-world probabilities.

A high score on a historical dataset does not prove reliable prediction of future landslides. Field validation and appropriate safety review are required before any real-world warning use.

## Future Improvements

- Collect and validate more field sensor data.
- Improve soil moisture calibration.
- Test under varying weather and ground conditions.
- Add a local display or alarm after appropriate safety validation.
- Implement packet-loss handling and receiver timeout detection.
- Develop a dashboard for monitoring sensor history and model outputs.

## License

Choose and add an appropriate open-source license before redistributing this project. Do not claim rights to third-party datasets or libraries that are subject to separate licenses.
