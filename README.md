# Smart-Waste-Classifier
# AI-Powered Waste Classification System (TFLite)

A lightweight and efficient computer vision application designed to automatically classify municipal waste into three primary categories: **Paper**, **Plastic**, and **Metal**.

This project utilizes an optimized **TensorFlow Lite (TFLite)** model to perform local, fast, and real-time image inference.

---

## How the System Works 

1. **Input:** The system takes a target image of waste (e.g., `TestImage.jpg`).
2. **Preprocessing:** The image is automatically resized to `224x224` pixels and normalized mathematically to match the model's input requirements.
3. **Inference:** The processed image is fed into the loaded TensorFlow Lite interpreter.
4. **Output:** The model outputs the predicted category (Paper, Plastic, or Metal) along with a confidence score representing its accuracy level.

---

##  Technologies Used 

* **Programming Language:** Python 3.13
* **Deep Learning Framework:** TensorFlow Lite (for executing the lightweight `.tflite` model)
* **Image Processing:** Pillow (PIL) (for image loading, resizing, and preprocessing)
* **Numerical Computation:** NumPy (for handling input image arrays and tensors)

---

## Project Structure

```text
├── MyCode.py             # Main inference script
├── model_unquant.tflite  # Optimized TFLite model 
├── labels.txt            # Classification classes (Paper, Plastic, Metal)
└── TestImage.jpg         # Sample test image


