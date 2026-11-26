# Can you fool the AI? 🕶️

[🇳🇱 Nederlands](README.md) · **🇬🇧 English** · [🌐 Website](https://tomjg14.github.io/computing-science-day-demo/?lang=en)

An interactive computer vision show for secondary school students, made for the Computing Science Day of Radboud University.
With a webcam and a big screen, it shows in four steps how a computer looks at images, how sure an AI is, and how you can fool an AI.

> ✉️ **Questions, or stuck with the installation?** Feel free to email me: [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl)

> 🔬 **The research behind the demo:** in 2016, researchers made glasses with a special pattern that made a face recognition system think the wearer was someone else.
> Read the paper: [Sharif et al. (2016), *Accessorize to a Crime*](https://doi.org/10.1145/2976749.2978392). More in [The research](#-the-research).

---

## 🎬 What is in the show?

| Step | What do you see? | What do you learn? |
|---|---|---|
| **Attract screen** | An invitation and today's leaderboard | – |
| **1. What does the computer see?** | A photo of your face turns into pixels, and every pixel turns into three numbers (red, green, blue). With **E**: how a filter finds edges, live (detailed: a real filter of the model on the pixels of the photo). | To a computer, a photo is just numbers. Filters turn them into clues, such as edges. |
| **2. How sure is the computer?** | Faces and objects with a confidence score. With four games: *Fool the AI!*, *What does the AI think?*, *Scavenger hunt* and *Confusion meter*. | An AI gives a score, not certainty, and it gets confused easily. |
| **3. Hide from the AI!** | Game: the AI sees your body but not your face = invisible. Easy (old AI from 2001) or hard (modern AI). With **E** first: why the AI can miss a face, with the live face score and measurements of your face. | Why an AI misses a face, and why the old AI is easier to fool than the modern one. |
| **4. How researchers do it** | A *simulation* of the glasses from the research, with an explanation of how such glasses are made. | Small changes in the right place can fool an AI. |

The show runs completely on your own computer. It stores no photos or videos (see [Privacy](#-privacy)).

---

## 🎓 Group lesson (20–30 students)

Next to the show for passers-by, the demo has a second variant: a lesson of about 20–25 minutes for one group.
The group stands in a half circle in front of the screen. A few volunteers stand in front and play; the whole group plays the group game.

```bash
python -m show --lesson
```

Or press **G** during the show (or the button in the panel) to switch. This does not change the show for passers-by.

| Chapter | What do you see? |
|---|---|
| **1. Opening** | The AI counts the group: "I see about 27 people". Everyone gets a thin, faint box; the players in front get a normal box. |
| **2. Pixels** | Step 1 of the show. Then **E**: how a filter finds edges (detailed: one real filter of the model on the pixels of the photo). |
| **3. How does the AI recognize things?** | Three explanation pages, live with real numbers: the old AI (2001), the modern AI (layers of filters), and how the clues become a box. **E** goes to the next page; after the last page, **E** closes them. **Page Down** shows the text one step at a time. |
| **4. How sure is the computer?** | Step 2 of the show, with the four games. *What does the AI think?* works well with the whole group. |
| **5. Hide from the AI!** | First the page *Why can the AI miss a face?* (**E** closes it), then step 3 of the show, with the volunteers in front. |
| **6. Hit the number!** | The group game (see below). |
| **7. The glasses** | Step 4 of the show (a *simulation*). |
| **8. Traffic sign** | Our own drawing of a stop sign with stickers: the same idea in another example. An *illustration*: no model looks at it. Source: Eykholt et al. (2018). |
| **9. Closing** | Thank you, and a large QR code. |

**Hit the number!** The AI must count exactly a target number of faces. The group decides who hides their face.

1. Everyone looks at the camera. Each face the AI sees gets a dot. **SPACE** measures the maximum. Faces that the camera cannot see do not count, so every target is possible.
2. The target starts at the maximum minus 1. Change it with **↑ / ↓** or in the panel.
3. The group discusses. Ready? **SPACE** starts the countdown. Then the AI counts for 2 seconds; the middle value counts, so one flicker does not decide.
4. The image freezes, with a dot on each counted face and the result. **SPACE**: next round. **BACKSPACE**: measure the maximum again.

The game keeps no scores. The lesson has no attract screen: it starts at the opening, and after the closing it goes back to the opening. When nobody is in view for a while, the lesson stays where it is.
Set the number of players in front (default 4) in the panel at *Max players*.

**Test without a group.** `python -m show --lesson --test` (or **Y**) adds a simulated group to the AI results: each person has a body and a face. **,** hides one face, **.** shows one again; the body stays, as in the real game. **Shift+,** / **Shift+.** removes or adds a whole person. A red label stays on screen as long as the test mode is on. The start size of the group is in `show/config.py` (`TEST_FACES`).

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

## 👩‍🏫 For the instructor

This section is for the person who gives the demo. It tells you which version to choose, what to prepare, and how to lead the explanations.

### Two versions

| | **Show for passers-by** | **Group lesson** |
|---|---|---|
| For | A stand at an open day: people walk by, look and play | One class of 20–30 students, about 20–25 minutes |
| Start | `python -m show` | `python -m show --lesson` |
| Flow | Attract screen → **SPACE** → steps 1–4 | Chapters 1–9, from the opening to the closing |
| Players | Up to 6 players in view | A few volunteers in front (default 4), plus group games for everybody |
| Nobody in view for 30 s | Back to the attract screen | Stays where it is |

Press **G** (or the button in the panel) to switch between the two versions while the show runs.

### Two levels of explanation

All explanation pages (key **E**) have two levels:

- **Simple** (default): one idea per page, in plain language, for students with no technical background.
- **Detailed**: more steps and the real numbers (weights, test values, the top 5 scores). Good for older students or a technical audience.

**Shift+E** (or *Level* in the panel) switches all explanations at once. The story is the same at both levels:

1. To a computer, a photo is only numbers (pixels).
2. Filters turn these numbers into clues, such as edges.
3. The old AI (2001) asks fixed yes/no questions and stops at the first NO. The modern AI learned its own filters, adds up many clues, and finds all objects in one look (YOLO).
4. The AI can miss a face when it gets too few clues.
5. With small changes in the right place, you can fool an AI. Researchers do this to make AI stronger and safer.

### Before you start

- **Camera and light:** the light falls on the faces, not from behind (no window behind the group). Check the image with **C** (next camera).
- **Screens:** the show on the big screen, the operator panel on your laptop (see [Setup](#️-setup-at-home-or-in-the-classroom)).
- **Language:** **L** switches the screen between Dutch and English.
- **Clicker:** most presentation clickers send *Page Down*, which shows the next text step. Test this first: some clickers send the arrow keys, and these change the step or chapter.
- **Group lesson:** set *Max players* in the panel, and tape a zone on the floor for the volunteers.
- **Practice alone:** add `--test` (or press **Y**) for a simulated group. See [Test without a group](#-group-lesson-2030-students).

### Lead the explanations

- **E** opens an explanation page. A page opens with only its title and the live image. The text waits for you.
- **Page Down** (or **N**, or the clicker) shows the next text step; **Page Up** goes one step back. Nothing moves on by itself, so you can wait for a good image, for attention, or for a question. The dots at the bottom show how many steps there are.
- The pages follow one face: the closest face, picked once. They never switch to another face by themselves. **TAB** picks the next face, for example when the volunteer changes.
- Everything on the explanation pages is live and comes from the real models. Parts that are only an illustration (the slowed-down search of the old AI, the glasses pattern) have a label on screen.

### Group lesson: chapter by chapter

| # | Chapter | What you do |
|---|---|---|
| 1 | Opening: "I see about N people" | Let the group come in and look at the count. |
| 2 | Pixels | **SPACE**: photo of a volunteer; **BACKSPACE**: live video again. **E**: how a filter finds edges (live: let the group wave). |
| 3 | How does the AI recognize things? | The old-AI page opens by itself. **E**: modern AI → from clues to a box → close (then free detection). |
| 4 | How sure is the computer? | **1** Fool, **2** Guess (whole group), **3** Hunt, **4** Confuse, **0** free. **I**: how-to-play card. |
| 5 | Why the AI misses + Hide from the AI! | The page *Why can the AI miss a face?* opens by itself: let a volunteer cover their face, turn and step back. **E** closes it. **SPACE**: start a round. **M**: easy / hard. |
| 6 | Hit the number! | **SPACE**: measure the maximum. **↑ / ↓**: target. **SPACE**: countdown. **SPACE**: next round. |
| 7 | The glasses (simulation) | **E**: how the glasses are made, and why researchers do this. |
| 8 | Traffic sign (illustration) | The same idea in another example. |
| 9 | Closing | The QR code to the website (**K**: on / off). |

### When something goes wrong

- **A volunteer does not become a player:** press **D** (debug). The panel shows the reason for each person. Often it helps to take one step closer, stand a bit apart, or look straight into the camera. Turn debug off again: its text also shows on the big screen.
- **The explanation follows the wrong face:** press **TAB** until the right face has the focus.
- **Step 3 does not work in this room (light, distance):** press **P** to jump to step 4.
- More problems and solutions: see [Troubleshooting](#-troubleshooting).

---

## 🎮 During the show

| Key | What does it do? |
|---|---|
| **SPACE** | Start the session (attract screen) or the action of the step: take a photo, start a round. During a round: stop it now and show the score (during the countdown: cancel the round). |
| **X** | During a round (step 2 game, step 3): 10 seconds more. Press again for more time. |
| **→ / ←** | Next / previous step. |
| **G** | Switch between the show for passers-by and the group lesson. |
| **Y** | Test mode on / off: a simulated group, to test the lesson without a group. **,** / **.**: hide one face / show one again. With **Shift**: remove one person / add one. |
| **↑ / ↓** | *Hit the number!*: target one higher / lower. |
| **BACKSPACE** | Step 1: back to the live video. *Hit the number!*: measure the maximum again. |
| **ESC** | Back to the attract screen (group lesson: the opening). With a leaderboard on screen, ESC only closes it. |
| **1 – 4 / 0** | Step 2: choose a game / back to free detection. |
| **I** | "How to play" card on / off. |
| **Shift+E** | All explanations: simple (default) / detailed. |
| **Page Down** / **N** | Explanation open: show the next text step (a presentation clicker sends Page Down). An explanation page opens with only its title and image; nothing moves on by itself. |
| **Page Up** | Explanation open: one text step back. |
| **TAB** | The explanations follow another face. They pick the closest face once and never switch by themselves. |
| **E** | Explanation, live with real numbers. Simple by default (for everyone); **Shift+E** switches all explanations to detailed. Step 1: how a filter finds edges (detailed: after the photo, one filter on your pixels). Step 2: old AI → modern AI → from clues to a box → off. Step 3: why the AI can miss a face. Step 4: how the glasses are made. |
| **M** | Step 3: easy (old AI) / hard (modern AI). |
| **V** | Scores on faces on / off (step 3: only during a round). |
| **T** | Step 2: next target (*Fool the AI!*, Shift+T: previous target) or choose one object (*Confusion meter*; after the last number back to automatic: all held objects count; you can also click an object in the panel). |
| **O** | Overlays on / off: title bar, info bars and QR code. Game meters and timers stay. |
| **K** | QR code on / off. |
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
- The show stores **no photos or videos**. The photo in step 1 exists only in memory and disappears when you move on. The same holds for the frozen image of *Hit the number!*: it disappears at the next round.
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
│   ├── players.py         who is playing, and why someone is not (yet) a player
│   ├── testmode.py        a simulated group, to test without a group (key Y)
│   ├── stages/            the steps: attract screen, 1 pixels, 2 detection, 3 hiding, 4 glasses,
│   │                      and the chapters of the group lesson (lesson.py, hitnumber.py)
│   └── games/             the four games of step 2
├── assets/                images: the glasses and the QR code
├── docs/                  the website (GitHub Pages)
├── tests/                 tests (game logic; the filter page computes as the model): python -m unittest discover -s tests -t .
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
| Someone in front does not become a player | Turn on debug (**D**). The panel and the screen show for each person why they are not a player. Often this helps: one step closer, stand a bit apart, look straight into the camera. If not, walk out of view and back in. |

---

## 🕰️ The previous version

The first version of this demo (with a Tkinter window and face recognition) is still in the project history, under the tag [`v1-oude-demo`](https://github.com/Tomjg14/computing-science-day-demo/tree/v1-oude-demo).

---

## ✉️ Contact

Questions? Send an email to [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl).
