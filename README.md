# Iris Recognition Assignment

**Author:** Ana Poklukar  
**Date:** November 2024

---

#### Overview
This assignment, part of the Image-Based Biometry class at the University of Ljubljana, focuses on iris recognition. The project leverages the [Open Source Iris Recognition](https://github.com/CVRL/OpenSourceIrisRecognition) library, specifically extending the HDBIF method, to implement custom feature extraction and matching techniques. The goal was to achieve accurate iris recognition by optimizing parameters and enhancing the existing pipeline.

The GitHub repository contains a Python implementation script for feature extraction and matching.

---

#### Features Implemented
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   - Enhances local contrast in polar-transformed iris images to improve feature extraction.

2. **Dual Circle Code Extraction:**
   - Binary codes are extracted around two concentric circles:
     - Circle 1: 8 points (P=8) on a radius of 1 pixel (R=1).
     - Circle 2: 16 points (P=16) on a radius of 2 pixels (R=2).

3. **Bilinear Interpolation:**
   - Ensures accurate pixel value sampling along the circles.

4. **Rotation Invariance:**
   - Normalizes binary codes by adjusting them to their lexicographically smallest rotation.

5. **Local Variance Calculation:**
   - Computes normalized local pixel variance for Circle 1 as an additional feature.

6. **Matching and Scoring:**
   - Compares binary codes using masked Hamming distances.
   - Combines Hamming distances from both circles (equal weights) with variance differences.
   - Computes a final similarity score using weighted components:
     - 0.8 for combined Hamming distance.
     - 0.2 for variance similarity.
