# Real-time YOLO Computer Vision Demonstration

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

## 🖥️ User Interface Guide

<img width="945" height="912" alt="ui" src="https://github.com/user-attachments/assets/40986449-585c-43df-85ff-2e114bd9b8d3" />

The control panel is designed for easy interaction during live demos:

- **Object Detection**:
  - **Add Object**: Type an object name (e.g., "keyboard") to start detecting it.
  - **Quick Add**: One-click buttons for common objects like Mobile, Cup, Bottle, and Laptop.
  - **Manage Objects**: View and remove currently active detection targets.
- **Face Recognition**:
  - **Capture & Name Face**: Freezes the current frame to let you label and save a detected face.
  - **Manage Database**: View list of known faces or clear the database.
- **System Controls**:
  - **Start/Stop Recording**: Saves the live feed to the `recordings/` folder.
  - **Switch Camera**: Cycle through available webcams.
  - **Shortcuts**: Press `[SPACE]` to toggle digital glasses, or `[F]` for fullscreen mode.

## 🧠 Background & Techniques

This demo integrates several key concepts in modern Computer Vision:

- **Object Detection (YOLO)**:
  Unlike older methods that scan images with a sliding window, [**YOLO (You Only Look Once)**](https://arxiv.org/abs/1506.02640) processes the entire image in a single forward pass of a neural network. It divides the image into a grid, where each cell is responsible for predicting bounding boxes and class probabilities if an object's center falls within it. This "single-shot" approach allows the model to see global context, resulting in extremely fast, real-time inference suitable for video streams.

- **Face Recognition**:
  The system makes use of the python library [_face_recognition_](https://pypi.org/project/face-recognition/) that utilizes a deep neural network to map facial features into a **128-dimensional vector space** (embedding). This network is typically trained using _Triplet Loss_, which forces the embeddings of the same person to be mathematically close, while pushing different people apart. To recognize a user, we calculate the Euclidean distance between the live face's embedding and the stored embeddings in our database; a distance below a specific threshold (e.g., 0.6) indicates a match.

- **Adversarial Attacks (Physical Glasses)**:
  The "Digital Glasses" mode is inspired by the paper _'Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition'_ by [Sharif et al. (2016)](https://dl.acm.org/doi/abs/10.1145/2976749.2978392). They demonstrated that neural networks are highly sensitive to specific pixel patterns (perturbations) that are imperceptible or benign to humans. By optimizing the texture on a pair of eyeglass frames, they could physically realize an attack that causes a face recognition system to misclassify the wearer (impersonation) or fail to detect them entirely (evasion). Our demo simulates this by overlaying a digital patch that disrupts the model's classification logic.

## 📦 Required Hardware

To run this demo, you will need:

1.  **Laptop or Desktop PC**:
    - Windows, macOS, or Linux.
    - A dedicated GPU (NVIDIA) is recommended for smoother performance but not required.
2.  **Webcam**:
    - Built-in laptop camera or external USB webcam.

## 🛠️ Installation

### 🐣 Prerequisites for Beginners

_(Disclaimer: If you are already familiar with Git, Python, and Conda, feel free to skip this section and proceed to step 1.)_

To run this demo, you need to install a few tools:

1.  **Git** (Version Control)
    - **What is it?**: Git is a tool that allows you to download code from the internet and manage versions. We use it to "clone" (download) this project.
    - **How to install**: Go to [git-scm.com/downloads](https://git-scm.com/downloads) and download the installer for your system. Run it and use the default settings.

2.  **Miniconda** (Python Manager)
    - **What is Python?**: Python is the programming language used to build this application.
    - **What is Miniconda?**: Managing Python versions and libraries can be tricky. Miniconda is a lightweight tool that creates isolated "environments" for your projects. This ensures that the tools needed for this demo don't interfere with other software on your computer.
    - **How to install**: Go to the [Miniconda Installer page](https://docs.anaconda.com/miniconda/install/). Download the installer for your system.
    - **How to use it**: After installation, you will use a special terminal:
      - **Windows**: Search for "Anaconda Prompt" or "Miniconda Prompt" in your Start Menu and open it.
      - **macOS/Linux**: Open your standard Terminal.

---

### ⚙️ Step-by-Step Setup

1.  **Get the Code**
    Open your terminal (Anaconda Prompt on Windows) and run:

    ```bash
    git clone https://github.com/Tomjg14/computing-science-day-demo.git
    cd computing-science-day-demo
    ```

2.  **Create the Environment**
    We will create a specific box (environment) for this project named `cv-demo` with Python 3.10 installed inside it.

    ```bash
    conda create -n cv-demo python=3.10
    ```

    _Type `y` and press Enter if asked to proceed._

    Now, "activate" (enter) that environment:

    ```bash
    conda activate cv-demo
    ```

    _(You should see `(cv-demo)` appear at the start of your command line)._

3.  **System Prerequisites**:
    We recommend installing `cmake` and `dlib` via Conda to avoid compilation issues on all platforms.

4.  **Install System Libraries**
    This project uses `dlib` for face recognition. Conda makes this easy to install:

    ```bash

    **Platform Specific Notes**:
    - **Linux**: If you encounter build errors, ensure Xcode tools are installed: `xcode-select --install`
    - **Linux**: If you get GUI errors later, install Tkinter: `sudo apt-get install python3-tk`
    - **macOS**: If you get build errors, install Xcode tools: `xcode-select --install`

    ```

5.  **Install Python Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    _Note: If you have a dedicated GPU, ensure you have the correct CUDA version of PyTorch installed._

    > **Note for GPU Users**: If you have a dedicated NVIDIA GPU, you may want to install the specific CUDA version of PyTorch for faster performance.

## 💻 Usage

Run the main script to start the application:

```bash
conda activate cv-demo
python main.py
```

## Demonstration

https://github.com/user-attachments/assets/7ac8ed9b-fa2a-4d28-b382-aa10f27e04bd

## Contact

Any questions, please send an email to: [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl)


Any questions, please send an email to: [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl)
