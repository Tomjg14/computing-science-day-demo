# Adversarial YOLO Demo

A real-time computer vision demonstration exploring object detection and adversarial attacks. This project uses **YOLOv8** to detect faces and objects via webcam, featuring a "digital adversarial glasses" mode that simulates how specific patterns can fool AI models into misclassifying a human face (e.g., as a "Panda").

## 🚀 Features

- **Real-Time Detection**:
  - **Face Detection**: Uses a specialized `yolov8-face` model for tight, accurate bounding boxes.
  - **Object Detection**: Detects 80+ standard COCO objects (e.g., cell phone, bottle, laptop).
- **Adversarial Mode**:
  - Press `SPACE` to toggle "Digital Glasses".
  - Overlays a pixelated mask on detected faces.
  - Demonstrates classification shifts (swaps label from "Face" to "Panda").
  - **Landmark Tracking**: Glasses automatically rotate and scale with your eyes using facial landmarks.
- **Smart Control Panel**:
  - GUI built with Tkinter.
  - **Smart Input**: Type "mobile" and it intelligently maps to "cell phone".
  - **Visual Feedback**: See active objects and system status.
- **Recording**: Capture your demo sessions to `.avi` files with a single click.
- **Customization**: Drop in your own `glasses.png` to test different patch designs.

## 🛠️ Installation

1.  **Clone the repository** (or download the files):

    ```bash
    git clone https://github.com/yourusername/adversarial-demo.git
    cd adversarial-demo
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment (conda or venv).

    ```bash
    pip install -r requirements.txt
    ```

    _Note: If you have a dedicated GPU, ensure you have the correct CUDA version of PyTorch installed._

## 💻 Usage

Run the main script to start the application:

```bash
python main.py
```
