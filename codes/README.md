# 🏖️ Automated Beach Sediment Grain Size Mapping System  

### **Overview**  
This project presents a **low-cost, camera-based automated mapping system** to estimate **sediment grain size distribution** and **classify beach types** (berm, intertidal, dune).  
It integrates **machine learning, IoT sensors, and GNSS/GPS positioning** for real-time, in-situ beach characterization — eliminating the need for tedious physical sampling and lab analysis.

---

## 🌍 **Problem Background**  
Traditional beach grain size measurement requires manual collection of sediment samples followed by laboratory analysis — a slow and repetitive process, especially since coastal sediments are dynamic and frequently change due to tides and wave actions.  
This system automates the process using **computer vision**, **edge computing**, and **drone-based coverage**.

---

## ⚙️ **System Architecture**

### **Hardware Components**
- **Camera Module** (enclosed in cone housing) → captures consistent, top-view sand images  
- **LED Light** → ensures uniform illumination  
- **Raspberry Pi / Jetson Nano** → performs edge inference  
- **GNSS/GPS Receiver** → logs precise coordinates of the measurement location  
- **Temperature & Humidity Sensors (DHT11/BME280)** → record environmental conditions  
- **Display Screen** (for handheld mode) → shows predictions locally  
- **Drone Mount (optional)** → enables autonomous aerial coverage of beach regions  

---

## 🧠 **Machine Learning Model**

### **Objective**  
Predict **grain size distribution** and **classify beach region type** from captured images.  

### **Pipeline**
1. **Data Collection**  
   - Collected sand surface images under varying lighting and weather conditions  
   - Annotated grain size distribution (e.g., fine, medium, coarse sand)  

2. **Preprocessing**
   - Image normalization & histogram equalization  
   - Noise removal using Gaussian blur  
   - Texture enhancement via Gabor and LBP features  

3. **Feature Extraction**
   - **CNN-based approach** (e.g., MobileNetV3 / EfficientNet)  
   - Extracts texture & color descriptors correlating to sediment size  

4. **Model Training**
   - **Techniques Used:**
     - Transfer Learning  
     - Data Augmentation (rotation, brightness adjustment)  
     - Stratified cross-validation  
   - **Frameworks:** TensorFlow / PyTorch  
   - **Evaluation Metrics:** Accuracy, F1-score, MSE (for regression-based size prediction)

5. **Inference on Edge Device**
   - Optimized and converted to **TensorFlow Lite / ONNX** for Raspberry Pi inference  
   - Real-time prediction with latency < 1s  

---

## ☁️ **Cloud & Database Integration**

- **Local Edge Storage:** temporary cache in Pi  
- **Cloud Database (Firebase / Supabase / AWS)** stores:
  - Image metadata  
  - Grain size prediction  
  - GPS coordinates  
  - Environmental data (temperature, humidity)  

- **Visualization Dashboard:**  
  Displays real-time geotagged map of beaches, grain size variation, and sensor readings.  

---

## 🚁 **Drone Integration**

- The ML module (without screen) can be attached to a drone.  
- Drone autonomously flies along predefined GPS waypoints.  
- Captures sand images and sends them to the onboard Pi for inference.  
- Sends results to the cloud after flight.  
- Enables study of **large or inaccessible coastal areas**.  

---

## 📈 **Validation**
- Physical sediment samples collected at one or more sites.  
- Lab-measured grain sizes compared with model predictions.  
- Correlation coefficient and RMSE used for validation accuracy.  

---

## 🧩 **Data Flow Summary**

| Source | Data | Process | Output |
|--------|------|----------|--------|
| Camera + Sensors | Image + Env Data | Edge ML Model | Grain Size + Beach Type |
| Raspberry Pi | Processed Data | Cloud Storage | Geo-tagged Dataset |
| Cloud Database | Records | Dashboard | Visual Analytics |

---

## 💡 **Key Features**
✅ Real-time grain size estimation  
✅ Low-cost, portable, and autonomous  
✅ Consistent lighting via LED ring  
✅ Edge inference (no network dependency)  
✅ GPS-tagged results with environmental data  
✅ Drone compatibility for wide-area mapping  

---

## 📊 **Tech Stack**

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Hardware** | Raspberry Pi, Camera, GPS, DHT11, LED, Drone |
| **ML Framework** | TensorFlow / PyTorch |
| **Backend / Cloud** | Firebase / Supabase / AWS |
| **Frontend Dashboard** | React / Streamlit / Plotly |
| **Data Storage** | PostgreSQL / Firestore |
| **Visualization** | Leaflet / Mapbox / Dash |

---

## 🧾 **Usage Instructions**

1. **Capture Sand Image:**  
   Device captures photo under LED light.  

2. **Run Model:**  
   Raspberry Pi performs real-time inference.  

3. **View Results:**  
   Grain size, GPS, temperature, and humidity appear on screen.  

4. **Upload Data:**  
   Data is automatically sent to cloud database.  

5. **Dashboard Access:**  
   Scientists can monitor and visualize results remotely.  

---

## 🧪 **Future Enhancements**
- Integrate hyperspectral imaging for improved granularity  
- Add auto-flight optimization via computer vision navigation  
- Use reinforcement learning for autonomous sampling strategies  
- Incorporate erosion prediction using temporal data  

---

## 👩‍🔬 **Team & Credits**
Developed by: **TopCoders**  
Institution / Organization: **Netaji Subhas University Of Technology**  
Project Type: **Applied ML + IoT + Remote Sensing**
