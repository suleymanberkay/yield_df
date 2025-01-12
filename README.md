# IoT-Integrated Agricultural Yield Prediction

VIDEO LINK:
https://aguedutr.sharepoint.com/:v:/s/asd/ESA0AzIrFm1LifmTWYKwA0QBuEJWuWbw4RGEgrkzw7_huw?e=rPvOb2&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D

### Overview
This project leverages IoT and machine learning to predict agricultural crop yields using real-time environmental data. By integrating an ESP32 microcontroller and a DHT11 sensor, the system collects temperature and humidity data, processes it using a trained TensorFlow Lite model, and provides yield predictions. The results are visualized on ThingSpeak for real-time monitoring.

---

### Key Features
- **Real-Time Data Collection**: Collects temperature and humidity data via IoT devices.
- **Accurate Predictions**: Uses a lightweight MLP model optimized for deployment on ESP32.
- **Visualization**: Displays predictions on ThingSpeak for easy monitoring.
- **Sustainability**: Promotes efficient resource utilization in agriculture.
- **Scalability**: Designed to adapt to various crops, regions, and environmental conditions.

---

### Setup Instructions

#### **1. Clone the Repository**
```bash
git clone https://github.com/suleymanberkay/yield_df.git
cd yield_df
```

#### **2. Install Python Dependencies**
Make sure you have Python 3.8 or higher installed. Run the following command to install the required libraries:

#### **3. Upload the Arduino Code**
1. Open the file `sketch_jan12a.ino` in the Arduino IDE.
2. Connect your ESP32 board to your computer.
3. Select the appropriate board (ESP32) and COM port in the Arduino IDE.
4. Upload the code to your ESP32 device.

#### **4. Run the Prediction Script**
1. Open the terminal in the project directory.
2. Start the Python script for predictions:
   ```bash
    python yield_df.py
    python app.py
    python tflite.py

   ```

#### **5. Monitor Outputs**
- **ThingSpeak Dashboard**: Visit your ThingSpeak channel to monitor real-time predictions.
- **Terminal Logs**: Observe sensor readings and yield predictions in the terminal.

---

### Hardware List
- **ESP32 Microcontroller**: Used for collecting and transmitting environmental data.
- **DHT11 Sensor**: Measures temperature and humidity.
- **Wi-Fi Module**: Integrated with ESP32 for internet connectivity.

---

### Usage
1. Power on the ESP32 and ensure the DHT11 sensor is connected.
2. Start the Python prediction script (`tflite.py`) on your computer.
3. The ESP32 collects sensor data and sends it to the Python server for processing.
4. The TensorFlow Lite model processes the data and returns the predicted yield.
5. Predictions are displayed on ThingSpeak for easy visualization.

---

### System Architecture
1. **Data Collection**:
   - The ESP32 microcontroller collects temperature and humidity data from the DHT11 sensor.
2. **Data Transmission**:
   - Sensor data is sent to the Python server via HTTP requests.
3. **Prediction**:
   - The TensorFlow Lite model (`model.tflite`) processes the incoming data.
   - Preprocessing is handled using the `preprocessor.joblib` pipeline.
4. **Visualization**:
   - Predictions are uploaded to ThingSpeak for remote monitoring.

---

### Example Usage
#### Input:
- **Temperature**: DT11 value
- **Rainfall**: DT11 value

#### Output:
- **Predicted Yield**: 85,432(example) kg/ha.

---

### Contributions
- **Machine Learning**:
  - Model training and optimization.
  - Conversion to TensorFlow Lite for deployment on ESP32.
- **IoT Integration**:
  - Configuring ESP32 and DHT11 for real-time data collection.
  - Communication with the prediction server via HTTP requests.
- **Deployment**:
  - Visualization of predictions on ThingSpeak.

---

### Future Enhancements
- Add additional sensors for more accurate environmental data (e.g., soil moisture).
- Expand dataset to include data from diverse regions and crops.
- Implement offline capabilities for regions with limited internet access.
- Integrate a user-friendly web or mobile application for easier monitoring.

---

### Contact
For questions or contributions, please contact **Berkay Yılmaz** at [yilmazsuleymanberkay14@gmail.com].
