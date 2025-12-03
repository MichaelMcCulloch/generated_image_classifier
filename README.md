# **Entropy Analysis: Thermodynamic Image Forensics**

A mechanistic forensic tool for distinguishing Authentic Photographs, AI-Generated Images (Deepfakes), and Synthetic Graphics based on Information Theory and Thermodynamics.  
Unlike typical AI detectors that use neural networks (which can be biased or fooled), this tool analyzes the **mathematical signatures of light physics** versus **generative algorithms**.

## **Core Principles**

This tool uses a **Dual-Domain Decision Tree** to classify images based on three "Old School" signal processing metrics:

1. **Gradient Kurtosis (Natural Statistics):**  
   * **Authentic Photos:** Heavy-tailed distribution (High Kurtosis, \>40). Real light creates extreme edges and vast flat areas.  
   * **AI Models:** Gaussian distribution (Low Kurtosis, \<20). Diffusion models tend to "smooth" the statistical outliers, creating rounder histograms.  
2. **LSB Correlation (Bit-Plane Physics):**  
   * **Synthetic/Code:** Perfect flat colors result in high correlation (\>0.6) in the Least Significant Bit.  
   * **AI/Photos:** Noise (thermal or generative) creates low correlation in the bit plane.  
3. **Neighbor Correlation (Micro-Thermodynamics):**  
   * **Authentic Sensors:** Photon shot noise is random. Pixel $i$ is largely independent of pixel $i+1$ in flat areas (Low Correlation).  
   * **AI Upsamplers:** Generative textures are interpolated from latent spaces. Pixel $i$ is mathematically linked to pixel $i+1$ (High Correlation).

## **Installation**

### **Prerequisites**

* Python 3.8+  
* PyTorch (CPU or CUDA)  
* Scipy, NumPy, Pillow

### **Setup**

\# Clone the repository  
git clone \[https://github.com/yourusername/entropy-analysis.git\](https://github.com/yourusername/entropy-analysis.git)  
cd entropy-analysis

\# Install dependencies  
pip install torch numpy scipy pillow

## **Usage**

You can scan a single image, a specific directory, or run the default scan on ./real and ./fake folders.

### **Scan Specific Image**

`uv run main.py path/to/image.jpg`

### **Scan Directory**

`uv run main.py path/to/dataset/`

### **Default Scan**

Checks for real/ and fake/ directories in the current location.  
`uv run main.py`

## **Interpreting Output**

The tool outputs a verdict along with the raw metrics used to make the decision.  
**Example 1: A Deepfake**  
fake/face.jpg: AI | Kurtosis=14.48 (Gaussian Texture)

*Reasoning:* The kurtosis is extremely low, indicating the image lacks the "spiky" edge statistics of a real camera.  
**Example 2: A Real Photo**  
real/face.jpg: Photographic | Kurtosis=74.10 (\>45)

*Reasoning:* High kurtosis indicates natural lighting and sensor physics.  
**Example 3: The "Gray Zone" (Hard Case)**  
fake/face2.jpg: AI | Kurtosis=31.24 (Gray Zone), NoiseCorr=0.335 (\>0.275)

*Reasoning:* The gradient stats (31.24) were ambiguous, but the high neighbor correlation (0.335) revealed the smoothness of an AI upsampler.

## **Limitations**

* **Strong JPEG Compression:** High compression can destroy the subtle noise signatures required for the "Gray Zone" tie-breaker.  
* **Film Grain Overlays:** Adversarial attacks that add heavy Gaussian noise can artificially lower the Neighbor Correlation, potentially fooling the thermodynamic check (though often raising Kurtosis suspiciously).