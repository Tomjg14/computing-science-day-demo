# Can you fool the AI? 🕶️

[🇳🇱 Nederlands](README.md) · **🇬🇧 English** · [🌐 Website](https://tomjg14.github.io/computing-science-day-demo/?lang=en)

An interactive computer vision show for secondary school students, made for the Computing Science Day of Radboud University.
With a webcam and a big screen, it shows in four steps how a computer looks at images, how sure an AI is, and how you can fool an AI.

> 🔬 **The research behind the demo:** in 2016, researchers made glasses with a special pattern that made a face recognition system think the wearer was someone else.
> Read the paper: [Sharif et al. (2016), *Accessorize to a Crime*](https://doi.org/10.1145/2976749.2978392). More in [The research](#-the-research).

---

## 🎬 What is in the show?

| Step | What do you see? | What do you learn? |
|---|---|---|
| **Attract screen** | An invitation and today's leaderboard | – |
| **1. What does the computer see?** | A photo of your face turns into pixels, and every pixel turns into three numbers (red, green, blue). | To a computer, a photo is just numbers. |
| **2. How sure is the computer?** | Faces and objects with a confidence score. With four games: *Fool the AI!*, *What does the AI think?*, *Scavenger hunt* and *Confusion meter*. | An AI gives a score, not certainty, and it gets confused easily. |
| **3. Hide from the AI!** | Game: the AI sees your body but not your face = invisible. Easy (old AI from 2001) or hard (modern AI). | Why the old AI is easier to fool than the modern one. |
| **4. How researchers do it** | A *simulation* of the glasses from the research, with an explanation of how such glasses are made. | Small changes in the right place can fool an AI. |

The show runs completely on your own computer. It stores no photos or videos (see [Privacy](#-privacy)).

---

## 🧰 What do you need?

- **A computer** with Windows, macOS or Linux. Tested on Windows; macOS and Linux should work, but are not tested.
  An NVIDIA graphics card makes the show smoother, but it is not necessary.
- **A webcam.** The built-in webcam of a laptop works. A separate USB webcam on top of a big screen works best.
- **One or two screens.** One laptop screen is enough. With a second screen (TV, monitor or projector), the audience sees the show big, and you keep the operator panel on your laptop.
- **Internet at the first start**, to download the AI models (about 60 MB). After that, the show works without internet.

---

## 🛠️ Installation (once)

New to Python? No problem: follow the steps exactly.

### 1. Install the tools

- **Miniconda** (Python with separate "environments", so this demo changes nothing in your other software):
  download it from [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/install/) and install it with the default settings.
- **Git** (optional, to download the code): [git-scm.com/downloads](https://git-scm.com/downloads).
  Without Git: on GitHub, click **Code → Download ZIP** and unzip the file.

### 2. Download the code

Open a terminal. On Windows: search the Start menu for **Anaconda Prompt** (or **Miniconda Prompt**).

```bash
git clone https://github.com/Tomjg14/computing-science-day-demo.git
cd computing-science-day-demo
```

Did you use the ZIP? Then use `cd` to go to the unzipped folder.

### 3. Make an environment and install the parts

```bash
conda create -n demo_day python=3.10
conda activate demo_day
pip install -r requirements.txt
```

You now see `(demo_day)` at the start of the line.

> **Do you have an NVIDIA graphics card?** Then first install PyTorch with GPU support from [pytorch.org](https://pytorch.org/get-started/locally/), and then `pip install -r requirements.txt`. Without this step the show also works, but slower.

---

## ▶️ Start the show

```bash
conda activate demo_day
python -m show
```

The first time, the show downloads the models; this takes a moment. Then two windows appear: **the show** and the **operator panel**.

Do you use a separate webcam, and does the show pick the wrong camera? Then start with the camera number, or press **C** during the show:

```bash
python -m show 1
```

---

## 🖥️ Setup at home or in the classroom

### With one screen (only a laptop)

1. Start the show. The show and the operator panel are next to each other on the same screen.
2. Control the show with the keys (see below). All keys work in both windows.
3. Do you want the show big? Press **F** for full screen. The panel is then behind the show; the keys still work. Press **F** again to go back.

### With two screens (laptop + TV, monitor or projector)

1. Connect the second screen and set it to **extend**, not duplicate.
   On Windows: press **Windows key + P** and choose **Extend**.
2. Start the show. The show goes full screen on the largest other screen by itself. The operator panel stays on your laptop.
3. Is the show on the wrong screen? Then set the screen number in `show/config.py` at `SHOW_DISPLAY` (show) and `PANEL_DISPLAY` (panel).

### The webcam

- **Built-in webcam:** works directly (camera 0). Place the laptop so that the audience is in view.
- **Separate USB webcam:** put it above or below the big screen, aimed at the audience. Start with `python -m show 1`, or press **C** until the right image shows.
- **Light:** make sure the light falls on the faces, not behind the people (no window behind them).
- **macOS:** give the Terminal access to the camera in *System Settings → Privacy & Security → Camera*.

---

## 🎮 During the show

| Key | What does it do? |
|---|---|
| **SPACE** | Start the session (attract screen) or the action of the step: take a photo, start a round. |
| **→ / ←** | Next / previous step. |
| **ESC** | Back to the attract screen. |
| **1 – 4 / 0** | Step 2: choose a game / back to free detection. |
| **I** | "How to play" card on / off. |
| **E** | Steps 3 and 4: extra explanation on / off (old vs. modern AI; how the glasses are made). |
| **M** | Step 3: easy (old AI) / hard (modern AI). |
| **V** | Scores on faces on / off (step 3: only during a round). |
| **T** | Step 2: other target (*Fool the AI!*) or other object (*Confusion meter*). |
| **P** | Jump to step 4 (backup when step 3 does not work). |
| **L** | Screen language: Dutch / English. |
| **C** | Next camera. |
| **F** | Full screen on / off. |
| **H** | All keys in the operator panel. |
| **Q** (2×) | Close the show. |

Every button in the operator panel does the same as a key. After a game round, press **N** to type a team name for the leaderboard.

### Settings

The most important settings are in [`show/config.py`](show/config.py), with an explanation for each. You can also change some of them in the panel during the show, for example how easily players become invisible in step 3.

---

## 🔒 Privacy

- The AI runs on your own computer: the camera images do not go to the internet. After the first download, the show needs no internet.
- The show turns off the anonymous usage statistics of Ultralytics.
- The show stores **no photos or videos**. The photo in step 1 exists only in memory and disappears when you move on.
- The leaderboard (`leaderboard.json`) keeps only team names and scores.

---

## 🔬 The research

### Glasses that fool an AI (step 4)

Step 4 is based on the work of Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer and Michael K. Reiter.
They showed that glasses with a computed pattern can fool a face recognition system: the AI no longer recognizes the wearer, or even sees the wearer as someone else.
This is how it works, in short:

1. To the AI, a face is numbers. It gives each possible answer a score.
2. The researchers change only the pixels of the glasses frame.
3. The math inside the AI tells them which change makes the wrong answer score higher. They repeat this very often.
4. They make the pattern smooth and printable, print the glasses and put them on: the attack also works in real life.

📄 **Read the papers:**

- M. Sharif, S. Bhagavatula, L. Bauer, M. K. Reiter (2016). *Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition.* ACM CCS 2016. [doi.org/10.1145/2976749.2978392](https://doi.org/10.1145/2976749.2978392)
- M. Sharif, S. Bhagavatula, L. Bauer, M. K. Reiter (2019). *A General Framework for Adversarial Examples with Objectives.* ACM Transactions on Privacy and Security. [doi.org/10.1145/3317611](https://doi.org/10.1145/3317611) · free version on [arXiv:1801.00349](https://arxiv.org/abs/1801.00349)

> ⚠️ **Note:** the glasses in step 4 are a **simulation**. The glasses are only drawn on the screen; the AI model does not see them, and the "Panda" label is fixed. The screen says so too.

### The two face detectors (step 3)

- **Old AI: Haar cascade** (Viola & Jones, 2001). A fixed series of light/dark tests on a gray image, such as "are the eyes darker than the cheeks?". The model in OpenCV has 25 checks in a row; if a window fails one check, there is no face. Cover your eyes or nose, and you are gone.
  P. Viola, M. Jones (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* CVPR 2001. [doi.org/10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)
- **Modern AI: YOLO** (neural network). YOLO ("You Only Look Once") looks at the whole image in one go and learned its own clues from very many example photos. Cover one part of your face, and other parts can still give you away.
  J. Redmon, S. Divvala, R. Girshick, A. Farhadi (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

The show uses the YOLO26 models of [Ultralytics](https://docs.ultralytics.com/) and the face model of [yolo-face](https://github.com/akanametov/yolo-face).

---

## 📁 How is the project built?

```text
computing-science-day-demo/
├── show/                  the show (start with: python -m show)
│   ├── app.py             main program: windows, keys, steps
│   ├── config.py          all settings, with explanations
│   ├── controls.py        list of all keys
│   ├── panel.py           the operator panel
│   ├── texts.py           all screen texts (Dutch and English)
│   ├── vision.py          camera and AI models
│   ├── players.py         who is playing (step 3)
│   ├── stages/            the steps: attract screen, 1 pixels, 2 detection, 3 hiding, 4 glasses
│   └── games/             the four games of step 2
├── assets/                images: the glasses and the QR code
├── docs/                  the website (GitHub Pages)
├── requirements.txt       the Python parts
├── README.md              this guide (Dutch)
└── README.en.md           this guide (English)
```

The AI models (`*.pt`) appear in the main folder by themselves at the first start.

---

## 🩺 Troubleshooting

| Problem | Solution |
|---|---|
| *"No working camera found"* | Close other programs that use the camera (Teams, Zoom). Try `python -m show 1`. On macOS: give the Terminal access to the camera. |
| The show is slow | Look at *Speed* in the operator panel. Close other programs. With an NVIDIA graphics card: install PyTorch with GPU support (see installation). |
| The operator panel is gone | It is behind the show (full screen). Press **F**, or just use the keys. |
| The model download fails | Check the internet connection and start again. |
| Wrong screen | Set `SHOW_DISPLAY` and `PANEL_DISPLAY` in `show/config.py`. |

---

## 🕰️ The previous version

The first version of this demo (with a Tkinter window and face recognition) is still in the project history, under the tag [`v1-oude-demo`](https://github.com/Tomjg14/computing-science-day-demo/tree/v1-oude-demo).

---

## ✉️ Contact

Questions? Send an email to [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl).
