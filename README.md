# Lane Detection & Control Pipeline 🚗

This project implements a full pipeline for detecting road lane lines and applying control logic using Python and OpenCV. It supports camera calibration, binary image thresholding, bird's-eye perspective transformation, polynomial lane fitting, and optionally applies control via PID, Kalman Filter, or Model Predictive Control (MPC).

---

## 🎬 Overview

[▶️ Watch overview video](data/overview.mp4)
<video width="640" height="360" controls>
  <source src="https://raw.githubusercontent.com/baolongdev/CarRacing2025_LaneDetection/main/data/overview.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## 📁 Project Structure

```
CarRacing2025/
├── main.py                    # Main script to run full pipeline
├── config.py                  # Configuration settings
├── create_project_structure.py# Auto-create folder structure
├── client_lib.so              # Compiled library (likely for simulator/client)
├── README.md                  # Project documentation
│
├── data/
│   ├── dataset_loader.py      # Load training or evaluation datasets
│   └── __init__.py
│
├── images/                    # Input or test images
│
├── logs/
│   └── log.txt                # Execution logs
│
├── models/
│   ├── Controller.py          # Top-level vehicle controller
│   ├── KalmanFilter.py        # Kalman Filter for state estimation
│   ├── MPC.py                 # Model Predictive Control for steering/throttle
│   ├── PIDController.py       # Classical PID controller
│   ├── SimulatorClient.py     # Interface to external simulator
│   └── __init__.py
│
├── pipeline/
│   ├── BirdEyeTransformer.py  # Perspective transform to bird's-eye view
│   ├── CameraCalibrator.py    # Camera calibration from chessboard
│   ├── ImageBinarizer.py      # Binary thresholding of lane
│   ├── LaneDetector.py        # Lane fitting and tracking logic
│   └── __init__.py
│
├── utils/
│   └── __init__.py            # Utility functions (if any)
```

---

## ✅ Features

- 📷 **Camera Calibration**  
  Computes distortion matrix from chessboard images.

- 🧼 **Image Binarization**  
  Uses Sobel filters and color thresholds to isolate lane lines.

- 👁 **Bird's Eye View**  
  Applies perspective transform to view road from top-down.

- 🧠 **Lane Tracking**  
  Supports both sliding window search and previous-frame polynomial reuse.

- 🧭 **Vehicle Control (MPC/PID/Kalman)**  
  Optional integration with controllers to close feedback loop in simulation.

- 🎨 **Visualization**  
  Draws lane overlay and curvature estimation on original image.

---

## 🔧 Installation

Make sure Python 3.7+ is installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
numpy
opencv-python
matplotlib
```

---

## ▶️ Running the Code

To process test images or integrate with simulator:

```bash
python main.py
```

---

## 📌 Notes

- Calibration data is cached automatically after first use.
- Perspective transform points assume 1280x720 images unless overridden.
- The `Line` class keeps recent polynomial fits and curvature estimates.
- Verbose mode in `main.py` will show debug plots and visualizations.

---

## 📷 Output Example

Final output includes:
- Green polygon showing lane area
- Red (left) and blue (right) lane lines
- Lane curvature and center offset estimation

---

## 🤝 Credits

- OpenCV, NumPy
- Inspired by Udacity's Self-Driving Car Nanodegree

---

## 🛠 Future Improvements

- Real-time camera input or video stream
- Lane departure warning
- Smoother steering integration via MPC
- ROS or Carla integration

---

## 📬 Maintainer

**Lê Bảo Long**