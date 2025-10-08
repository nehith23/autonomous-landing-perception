# Computer Vision and Sensing Techniques for Autonomous Drone Landing in Space Exploration

This repository contains the coursework project for **COMP0241 –  Computer Vision and Sensing**, focusing on developing and evaluating computer vision algorithms for **autonomous drone landing and navigation in space environments**.  
The project integrates segmentation, motion tracking, and rotation analysis to estimate geometric and dynamic properties of a suspended Astronomical Object (AO).

![ORB_Matching](https://github.com/user-attachments/assets/a1536f23-f220-435b-ad71-cf89fd6a198b)

---

## Overview
The project develops a perception pipeline that combines **vision-based sensing** and **geometric analysis** to extract motion cues, surface velocity, and object rotation information. These insights guide safe and precise drone landing strategies in unknown or dynamic space-like environments.

---

## Objectives
- Segment and isolate the Astronomical Object (AO) in diverse visual conditions.  
- Estimate AO geometry: center, diameter, and height above the ground.  
- Compute the AO’s rotation cycle from video data.  
- Derive surface velocity parameters to inform drone navigation and control.

---

## Features
- **Image Segmentation:** HSV and Hough Transform-based methods, combined for accuracy.  
- **Geometric Analysis:** Centroid tracking and stereo-based height estimation.  
- **Rotation Cycle Estimation:** ORB + RANSAC feature matching for temporal motion analysis.  
- **Drone Navigation Parameters:** Computation of diameter and surface velocities as a function of latitude.

---

## Methodology

### Task 1: Image Segmentation
- **Techniques:** HSV color thresholding, Hough Transform, combined segmentation.  
- **Evaluation:** ROC curves, IoU, F1-score, and AUC metrics.  
<img width="480" alt="Segmentation Results" src="https://github.com/user-attachments/assets/2e5afa63-126f-425f-a132-f8c98a6e1d1e" />

### Task 2: Geometric Analysis
- **Center Tracking:** Extracted AO centroid; sinusoidal motion patterns observed.  
- **Height Estimation:** Stereo depth computation with calibrated cameras.  
<img width="640" alt="Geometric Analysis" src="https://github.com/user-attachments/assets/5030fb92-afca-47d1-8340-fb5401cf710e" />

### Task 3: Rotation Cycle Estimation
- **Approach:** ORB feature matching + RANSAC filtering for peak rotation detection.  
- **Result:** Reliable cycle estimation under noisy, real-time conditions.  
<img width="785" alt="Rotation Estimation" src="https://github.com/user-attachments/assets/e96bc24c-79cd-4abc-adb1-0cd08ea31851" />

---

## Tools and Libraries
- **Language:** Python 3.11  
- **Core Libraries:**  
  - OpenCV – Image processing and feature matching  
  - NumPy – Numerical computations  
  - Matplotlib – Data visualization  
  - scikit-learn – Evaluation metrics  
  - PyAutoGUI – Frame capture and automation

---

## Results
- Achieved high segmentation accuracy with strong IoU and AUC metrics.  
- Extracted precise geometric and motion parameters.  
- Robust rotation cycle estimation even under lighting and motion noise.  
- Derived actionable surface velocity parameters for navigation control.

---

## Challenges
- Inconsistent lighting and background interference reduced segmentation quality.  
- Irregular AO motion introduced centroid tracking noise.  
- Real-time limitations mitigated through adaptive frame selection and filtering.

---

## Future Work
- Integrate deep learning models (U-Net, Mask R-CNN) for adaptive segmentation.  
- Optimize the full pipeline for real-time inference using GPU acceleration.  
- Extend methodology to handle non-spherical or irregular rotating bodies.

---

## Contact
**GitHub:** [nehith23](https://github.com/nehith23)  
**LinkedIn:** [Nehith V](https://www.linkedin.com/in/nehith-v)  
**Email:** ucabvem@ucl.ac.uk
