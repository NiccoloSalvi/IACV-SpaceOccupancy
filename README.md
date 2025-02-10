# 🚀 Car Tracking in the Dark

## 📌 Overview
This project focuses on **space occupancy of a moving vehicle** using video analysis from a fixed camera. The system determines the occupied space for each frame, leveraging intrinsic camera parameters and a simplified car model.

## 🎯 Purpose
The goal of this system is to:
- Analyze video footage to determine the space occupied by a moving vehicle.
- Utilize calibration parameters and a simplified car model.

## 📋 Assumptions
- The car has a **vertical symmetry plane**.
- Two symmetric rear lights are visible.
- The camera observes the car from the **back**.
- The car is either **translating forward** or **steering with constant curvature**.
- The road is **locally planar**.

## 🛠 System Operation
### 🔧 Offline Steps
- **Camera Calibration**
- **Retrieve Car Model**
  - Width, length, rear light positions, and their height above ground

### ⚙️ Online Steps (Frame-Based Processing)
1. Extract **two symmetric lights** from consecutive frames.
2. Define the **light segment**, the horizontal segment joining the lights.
3. Apply geometric transformations to determine the **3D position** of the segment.
4. Use the **car model** and computed position to estimate the space occupied on the road.

## 📐 Geometric Processing
### 🚗 Case 1: Car Moving Forward
- The light segments in consecutive frames form a **rectangle**.
- The **center of rotation** $C$ is determined.
- **Parallel line constraints** help identify motion characteristics.

### 🔄 Case 2: Car Steering with Constant Curvature
- The light segment **rotates** about a center of rotation $C$.
- A **bisecting line** $b$ aids in motion analysis.

## 📏 Methods Used
### **1️⃣ Translating Forward**
- Identify intersection points $V_x$ and $V_y$.
- Compute the **3D direction** of the light segment.
- Determine if the car is moving forward by checking perpendicularity.
- Estimate **vanishing lines** and compute the **camera-to-plane distance**.

### **2️⃣ Steering with Constant Curvature**
- Find intersection points $C_b$ and $V_y$.
- Validate if **directional constraints** hold.
- Compute **vanishing lines** and estimate distances.

## ⚠️ Limitations & Recommendations
### 🔴 Limitations
- Requires **sufficient perspective** to ensure accurate calculations.
- Parallel viewing rays may require additional **symmetry-based methods**.

### ✅ Recommendations
- Position the **camera at an inclined angle** to ensure **better perspective**.
- Focus on the **straightforward translation** case first.
- Utilize **ground truth data** from a reference plane for verification.

## 📍 Car Localization Approaches
### 🔹 1. **Standard Localization**
- Uses a **single image** and **homography estimation**:
  $$\mathbf{r}_1 \mathbf{r}_2 \mathbf{o}_{\pi} = K^{-1} H$$

### 🔹 2. **Nighttime Localization (Image Pair-Based)**
- Uses two **non-consecutive** frames to increase perspective.
- Extracts symmetric features and applies **geometric processing**.

### 🔹 3. **Nighttime Localization (Symmetric Elements-Based)**
- Uses multiple symmetric elements to **estimate rotation**.

## 📏 Computing Horizontal Inclination Angle $\theta$
- Extracts **y-coordinates** from key points.
- Uses CAD model data to compute **rotation**.
- Estimates **3D position** based on segment lengths.

## 🎥 Experimentation & Ground Truth
- A **daytime video** can serve as a reference **ground truth**.
- Once **3D positions** of car elements are known, a **bounding box** defines the **occupied space**.

## 📂 Repository Structure
```
📂 car-space-occupancy
 ├── 📜 README.md    # This file
 ├── 📜 assignment.pdf    # Assignment document
 ├── 📂 src          # Source code
 ├── 📂 data         # Video datasets
 ├── 📂 docs         # Documentation
```

## 🚀 Getting Started
1. **Clone the Repository**
   ```sh
   git clone https://github.com/NiccoloSalvi/IACV-SpaceOccupancy.git
   ```
2. **Navigate to the Directory**
   ```sh
   cd IACV-SpaceOccupancy
   ```
3. **Run the Application**
   ```sh
   python main.py
   ```

## 📜 License
This project is licensed under the **MIT License**.