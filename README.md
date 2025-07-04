# 🚗 Car Space Occupancy - IACV Project 2024-25

## 📌 Overview

This project addresses the specific challenge of developing a vision-based system capable of analyzing videos of moving vehicles captured by a fixed camera to draw a rectangular parallelepiped bounding box around the vehicle for each frame under low-light conditions. The primary objective is not to determine the actual space occupied by the car, but rather to accurately estimate and visualize the 3D bounding volume that encompasses the entire vehicle using geometric reconstruction techniques. The system must operate with minimal prior information, requiring only the intrinsic calibration parameters of the camera (K matrix) and a simplified model of the observed vehicle including basic dimensional parameters such as length, width, and the spatial configuration of rear lights.

---

## 📋 Assumptions

* The car has a **vertical symmetry plane**.
* Two **symmetric rear lights** are visible.
* The camera observes the car from **behind**.
* The vehicle is either **translating forward** or **steering at constant curvature**.
* The road is **locally planar**.

---

## 🧠 Methods Implemented

### **method1**: Standard Localization via Homography

→ Uses four coplanar points on the rear facade of the car (e.g., rear lights and license plate corners) to estimate pose from a single image using homography decomposition.

### **method2**: Nighttime Localization from Image Pairs

→ Designed for low-light settings, this method uses symmetric rear light points tracked across two frames. It exploits temporal motion cues to estimate the pose of the vehicle on the road.

### **method3**: Localization under Poor Perspective using Out-of-Plane Symmetric Features

→ When perspective cues are weak, this method uses symmetric features located on different planes of the car’s 3D structure (e.g., lights and mirrors) to recover pose by exploiting inter-frame symmetry and vanishing geometry.

### **method4**: PnP-based Vehicle Pose Estimation from Key Points

→ This method estimates the vehicle’s 3D pose from a single image by solving the Perspective-n-Point (PnP) problem using a set of known 3D key points (e.g., rear lights, license plate corners, and side mirror) and their 2D projections. By including non-coplanar points such as the side mirror, the method improves robustness and accuracy, especially in low-perspective conditions.

Each method folder contains:

* `main.py`: Run the method from the root with `python methodX/main.py`.
* `results/`: Output bounding boxes and overlays.

---

## 📂 Repository Structure

```
📁 IACV-SpaceOccupancy
├── 📁 cameraCalibration        # Intrinsics + distortion estimation
├── 📁 featureExtraction        # Frame sampling, light segmentation
├── 📁 method1                  # Homography-based localization
├── 📁 method2                  # Nighttime 3D triangulation (main method)
├── 📁 method3                  # Constant curvature steering case
├── 📁 method4                  # Symmetry and weak perspective cases
├── 📜 assignment.pdf           # Official assignment given by the professor
├── 📜 CAD_model.png            # Reference CAD sketch of car
├── 📜 video.mp4                 # Original video file
├── 📜 LICENSE                  # MIT license
├── 📜 README.md                # You're reading it!
```

---

## 🚀 Getting Started

1. **Clone the Repository**

```bash
git clone https://github.com/NiccoloSalvi/IACV-SpaceOccupancy.git
```

2. **Navigate to the Project Root**

```bash
cd IACV-SpaceOccupancy
```

3. **Run One of the Methods**

```bash
python method2/main.py
```

> ℹ️ You can replace `method2` with `method1`, `method3`, or `method4` to try different strategies.

---

## 📜 License

This project is licensed under the **MIT License**.
See `LICENSE` for details.

---

## 👥 Authors & Contributors

This project was developed as part of the **Image Analysis and Computer Vision** course at Politecnico di Milano.
It has been created in collaboration by:

* [@alessiovilla](https://github.com/alessiovilla)
* [@beazani](https://github.com/beazani)

---
