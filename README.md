# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## SEMICON Hackathon 2026 – Problem Statement 1

### Overview

Semiconductor manufacturing relies on microscopic inspection images to measure, verify, and detect defects in chips at different stages of production. These images need to be sharp and clean, as even small amounts of noise or loss of detail can hide critical defects.

This project uses an AI-based image restoration model to improve degraded semiconductor inspection images by addressing:

* **Speckle Noise** – Random pixel-level noise that produces a grainy appearance and can distort true pixel values.
* **Spatial Resolution Reduction** – Loss of fine image details caused by downsampling.

The developed model performs image restoration and **2× super-resolution**, converting degraded lower-resolution images into cleaner, higher-resolution outputs.

---

## Model Architecture

The project uses a custom **SRUNet (Super-Resolution U-Net)** implemented using PyTorch.

The architecture consists of:

1. **Encoder**

   * Extracts important features from the degraded input image.
   * Uses convolutional layers, batch normalization, ReLU activation, and max pooling.

2. **Bottleneck**

   * Processes the deepest feature representation.

3. **Decoder**

   * Reconstructs the image using transposed convolution layers.
   * Uses skip connections to preserve important spatial information.

4. **Super-Resolution Head**

   * Performs final 2× upscaling.
   * For example, a 128×128 input can produce a 256×256 output.

---

## Repository Structure

```text
semicon-hackathon-ps1/
│
├── model.py
│   └── Defines the SRUNet model architecture
│
├── run.py
│   └── Runs batch inference on degraded input images
│
├── requirements.txt
│   └── Required Python dependencies
│
└── models/
    └── srunet_overnight_weights.pth
        └── Trained model weights
```

---

## Requirements

The project requires:

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Model

The inference script accepts an input directory containing `.npy` image files and an output directory.

Run:

```bash
python run.py <input-dir> <output-dir>
```

### Example

```bash
python run.py input_images restored_images
```

The script will:

1. Load the trained SRUNet model.
2. Load the trained weights from:

```text
models/srunet_overnight_weights.pth
```

3. Read all `.npy` files from the input directory.
4. Process each degraded image using the trained model.
5. Restore and super-resolve the image.
6. Remove invalid NaN/Infinity values.
7. Clamp output pixel values to the range `[0, 1]`.
8. Save the restored output with the same filename in the output directory.

---

## Input and Output

### Input

* Degraded semiconductor inspection images.
* Stored as `.npy` files.
* Single-channel image data.

### Output

* AI-restored images.
* Reduced noise and improved spatial resolution.
* Pixel values constrained between `0` and `1`.
* Saved as `.npy` files.

---

## Objective

The objective of this project is to use AI-based image restoration to improve the quality of degraded semiconductor inspection images.

By reducing noise and recovering lost spatial detail, the system aims to produce clearer images that can support more accurate semiconductor inspection and defect analysis.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Torchvision
* Matplotlib

---

## Team

Developed for **SEMICON Hackathon 2026 – Problem Statement 1**.
