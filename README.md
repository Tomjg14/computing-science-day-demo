# Kun jij de AI foppen? 🕶️

**🇳🇱 Nederlands** · [🇬🇧 English](README.en.md) · [🌐 Website](https://tomjg14.github.io/computing-science-day-demo/)

Een interactieve show over computer vision voor scholieren, gemaakt voor de Computing Science Day van de Radboud Universiteit.
Een webcam en een groot scherm laten in vier stappen zien hoe een computer naar beelden kijkt, hoe zeker een AI is, en hoe je een AI kunt foppen.

> 🔬 **Het onderzoek achter de demo:** onderzoekers maakten in 2016 een bril met een speciaal patroon die een gezichtsherkenner liet denken dat de drager iemand anders was.
> Lees het artikel: [Sharif et al. (2016), *Accessorize to a Crime*](https://doi.org/10.1145/2976749.2978392). Meer in [Het onderzoek](#-het-onderzoek).

---

## 🎬 Wat zit er in de show?

| Stap | Wat zie je? | Wat leer je? |
|---|---|---|
| **Wachtscherm** | Uitnodiging en het scorebord van vandaag | – |
| **1. Wat ziet de computer?** | Een foto van je gezicht wordt pixels, en elke pixel wordt drie getallen (rood, groen, blauw). | Voor een computer is een foto alleen getallen. |
| **2. Hoe zeker is de computer?** | Gezichten en voorwerpen met een zekerheidsscore. Met vier spellen: *Bedrieg de AI!*, *Wat denkt de AI?*, *Speurtocht* en *Verwarringsmeter*. | Een AI geeft een score, geen zekerheid, en raakt snel in de war. |
| **3. Verstop je voor de AI!** | Spel: de AI ziet je lichaam, maar niet je gezicht = onzichtbaar. Makkelijk (oude AI uit 2001) of moeilijk (moderne AI). | Waarom de oude AI makkelijker te foppen is dan de moderne. |
| **4. Zo doen onderzoekers het** | Een *simulatie* van de bril uit het onderzoek, met uitleg hoe zo'n bril gemaakt wordt. | Kleine veranderingen op de juiste plek kunnen een AI foppen. |

De show draait helemaal op je eigen computer. Hij slaat geen foto's of video's op (zie [Privacy](#-privacy)).

---

## 🧰 Wat heb je nodig?

- **Een computer** met Windows, macOS of Linux. Getest op Windows; macOS en Linux zouden moeten werken, maar zijn niet getest.
  Een NVIDIA-videokaart maakt de show vlotter, maar is niet nodig.
- **Een webcam.** De ingebouwde webcam van een laptop werkt. Een losse USB-webcam boven op een groot scherm werkt het best.
- **Eén of twee schermen.** Eén laptopscherm is genoeg. Met een tweede scherm (tv, monitor of beamer) ziet het publiek de show groot, en houd jij het bedieningspaneel op je laptop.
- **Internet bij de eerste start**, om de AI-modellen te downloaden (ongeveer 60 MB). Daarna werkt de show zonder internet.

---

## 🛠️ Installatie (eenmalig)

Nieuw met Python? Geen probleem: volg de stappen precies.

### 1. Installeer de hulpprogramma's

- **Miniconda** (Python met aparte "omgevingen", zodat deze demo niets van je andere software verandert):
  download het via [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/install/) en installeer met de standaardinstellingen.
- **Git** (optioneel, om de code te downloaden): [git-scm.com/downloads](https://git-scm.com/downloads).
  Zonder Git kan het ook: klik op GitHub op **Code → Download ZIP** en pak het bestand uit.

### 2. Download de code

Open een terminal. Op Windows: zoek in het Startmenu naar **Anaconda Prompt** (of **Miniconda Prompt**).

```bash
git clone https://github.com/Tomjg14/computing-science-day-demo.git
cd computing-science-day-demo
```

Gebruik je de ZIP? Ga dan met `cd` naar de uitgepakte map.

### 3. Maak een omgeving en installeer de onderdelen

```bash
conda create -n demo_day python=3.10
conda activate demo_day
pip install -r requirements.txt
```

Je ziet nu `(demo_day)` aan het begin van de regel.

> **Heb je een NVIDIA-videokaart?** Installeer dan eerst PyTorch met GPU-ondersteuning via [pytorch.org](https://pytorch.org/get-started/locally/), en daarna `pip install -r requirements.txt`. Zonder deze stap werkt de show ook, maar langzamer.

---

## ▶️ De show starten

```bash
conda activate demo_day
python -m show
```

De eerste keer downloadt de show de modellen; dat duurt even. Daarna verschijnen twee vensters: **de show** en het **bedieningspaneel**.

Gebruik je een losse webcam en kiest de show de verkeerde camera? Start dan met het cameranummer, of druk tijdens de show op **C**:

```bash
python -m show 1
```

---

## 🖥️ Opstelling thuis of in de klas

### Met één scherm (alleen een laptop)

1. Start de show. De show en het bedieningspaneel staan naast elkaar op hetzelfde scherm.
2. Bedien de show met de toetsen (zie hieronder). Alle toetsen werken in beide vensters.
3. Wil je de show groot? Druk op **F** voor volledig scherm. Het paneel zit dan achter de show; de toetsen blijven werken. Druk nog een keer op **F** om terug te gaan.

### Met twee schermen (laptop + tv, monitor of beamer)

1. Sluit het tweede scherm aan en zet het op **uitbreiden**, niet op dupliceren.
   Op Windows: druk op **Windows-toets + P** en kies **Uitbreiden**.
2. Start de show. De show gaat vanzelf volledig scherm op het grootste andere scherm. Het bedieningspaneel blijft op je laptop.
3. Komt de show op het verkeerde scherm? Zet dan in `show/config.py` het schermnummer bij `SHOW_DISPLAY` (show) en `PANEL_DISPLAY` (paneel).

### De webcam

- **Ingebouwde webcam:** werkt direct (camera 0). Zet de laptop zo neer dat het publiek in beeld komt.
- **Losse USB-webcam:** zet hem boven of onder het grote scherm, gericht op het publiek. Start met `python -m show 1`, of druk op **C** tot het goede beeld komt.
- **Licht:** zorg voor licht op de gezichten, niet achter de mensen (geen raam in de rug).
- **macOS:** geef de Terminal toegang tot de camera via *Systeeminstellingen → Privacy en beveiliging → Camera*.

---

## 🎮 Tijdens de show

| Toets | Wat doet het? |
|---|---|
| **SPATIE** | Start de sessie (wachtscherm) of de actie van de stap: foto maken, ronde starten. |
| **→ / ←** | Volgende / vorige stap. |
| **ESC** | Terug naar het wachtscherm. |
| **1 – 4 / 0** | Stap 2: kies een spel / terug naar vrij kijken. |
| **I** | Uitleg "hoe speel je het" aan / uit. |
| **E** | Stap 3 en 4: extra uitleg aan / uit (oude vs. moderne AI; hoe de bril gemaakt wordt). |
| **M** | Stap 3: makkelijk (oude AI) / moeilijk (moderne AI). |
| **V** | Scores op gezichten aan / uit (stap 3: alleen tijdens een ronde). |
| **T** | Stap 2: ander doel (*Bedrieg de AI!*) of ander voorwerp (*Verwarringsmeter*). |
| **P** | Direct naar stap 4 (reserve als stap 3 niet lukt). |
| **L** | Taal van het scherm: Nederlands / Engels. |
| **C** | Volgende camera. |
| **F** | Volledig scherm aan / uit. |
| **H** | Alle toetsen in het bedieningspaneel. |
| **Q** (2×) | Show afsluiten. |

Elke knop in het bedieningspaneel doet hetzelfde als een toets. Na een spelronde kun je met **N** een teamnaam invullen voor het scorebord.

### Instellingen

De belangrijkste instellingen staan in [`show/config.py`](show/config.py), met uitleg erbij. Een paar kun je ook tijdens de show in het paneel aanpassen, bijvoorbeeld hoe makkelijk spelers in stap 3 onzichtbaar worden.

---

## 🔒 Privacy

- De AI rekent op je eigen computer: de camerabeelden gaan niet naar internet. Na de eerste download heeft de show geen internet meer nodig.
- De show zet de anonieme gebruiksstatistieken van Ultralytics uit.
- De show slaat **geen foto's of video's** op. De foto in stap 1 staat alleen in het geheugen en verdwijnt als je verdergaat.
- Het scorebord (`leaderboard.json`) bewaart alleen teamnamen en scores.

---

## 🔬 Het onderzoek

### De bril die een AI fopt (stap 4)

Stap 4 is gebaseerd op het werk van Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer en Michael K. Reiter.
Zij lieten zien dat een bril met een berekend patroon een gezichtsherkenner kan foppen: de AI herkent de drager niet meer, of ziet de drager zelfs als iemand anders.
Zo werkt het, in het kort:

1. Voor de AI is een gezicht getallen. Hij geeft elk mogelijk antwoord een score.
2. De onderzoekers veranderen alleen de pixels van het brilmontuur.
3. De rekenregels in de AI vertellen welke verandering de score van het foute antwoord verhoogt. Dat herhalen ze heel vaak.
4. Ze maken het patroon glad en printbaar, printen de bril en zetten hem op: de aanval werkt ook in het echt.

📄 **Lees de artikelen:**

- M. Sharif, S. Bhagavatula, L. Bauer, M. K. Reiter (2016). *Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition.* ACM CCS 2016. [doi.org/10.1145/2976749.2978392](https://doi.org/10.1145/2976749.2978392)
- M. Sharif, S. Bhagavatula, L. Bauer, M. K. Reiter (2019). *A General Framework for Adversarial Examples with Objectives.* ACM Transactions on Privacy and Security. [doi.org/10.1145/3317611](https://doi.org/10.1145/3317611) · gratis versie op [arXiv:1801.00349](https://arxiv.org/abs/1801.00349)

> ⚠️ **Let op:** de bril in stap 4 is een **simulatie**. De bril wordt alleen op het scherm getekend; het AI-model ziet hem niet, en het label "Panda" staat vast. Het scherm zegt dat ook.

### De twee gezichtsdetectoren (stap 3)

- **Oude AI: Haar-cascade** (Viola & Jones, 2001). Een vaste reeks licht/donker-testen op een grijs beeld, zoals "zijn de ogen donkerder dan de wangen?". Het model in OpenCV heeft 25 controles na elkaar; zakt een venster voor één controle, dan is er geen gezicht. Bedek je ogen of neus, en je bent weg.
  P. Viola, M. Jones (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* CVPR 2001. [doi.org/10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)
- **Moderne AI: YOLO** (neuraal netwerk). YOLO ("You Only Look Once") bekijkt het hele beeld in één keer en leerde zijn kenmerken zelf uit heel veel voorbeeldfoto's. Bedek je één deel van je gezicht, dan verraden andere delen je nog.
  J. Redmon, S. Divvala, R. Girshick, A. Farhadi (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

De show gebruikt de YOLO26-modellen van [Ultralytics](https://docs.ultralytics.com/) en het gezichtsmodel van [yolo-face](https://github.com/akanametov/yolo-face).

---

## 📁 Hoe zit het project in elkaar?

```text
computing-science-day-demo/
├── show/                  de show (start met: python -m show)
│   ├── app.py             hoofdprogramma: vensters, toetsen, stappen
│   ├── config.py          alle instellingen, met uitleg
│   ├── controls.py        lijst van alle toetsen
│   ├── panel.py           het bedieningspaneel
│   ├── texts.py           alle schermteksten (Nederlands en Engels)
│   ├── vision.py          camera en AI-modellen
│   ├── players.py         wie speelt er mee (stap 3)
│   ├── stages/            de stappen: wachtscherm, 1 pixels, 2 detectie, 3 verstoppen, 4 bril
│   └── games/             de vier spellen van stap 2
├── assets/                plaatjes: de bril en de QR-code
├── docs/                  de website (GitHub Pages)
├── requirements.txt       de Python-onderdelen
├── README.md              deze uitleg (Nederlands)
└── README.en.md           deze uitleg (Engels)
```

De AI-modellen (`*.pt`) komen bij de eerste start vanzelf in de hoofdmap.

---

## 🩺 Problemen oplossen

| Probleem | Oplossing |
|---|---|
| *"No working camera found"* | Sluit andere programma's die de camera gebruiken (Teams, Zoom). Probeer `python -m show 1`. Op macOS: geef de Terminal toegang tot de camera. |
| De show is traag | Kijk in het bedieningspaneel bij *Speed*. Sluit andere programma's. Met een NVIDIA-videokaart: installeer PyTorch met GPU-ondersteuning (zie installatie). |
| Het bedieningspaneel is weg | Het zit achter de show (volledig scherm). Druk op **F**, of gebruik gewoon de toetsen. |
| De download van de modellen mislukt | Controleer de internetverbinding en start opnieuw. |
| Verkeerd scherm | Zet `SHOW_DISPLAY` en `PANEL_DISPLAY` in `show/config.py`. |

---

## 🕰️ De vorige versie

De eerste versie van deze demo (met een Tkinter-venster en gezichtsherkenning) staat nog in de geschiedenis van het project, onder de tag [`v1-oude-demo`](https://github.com/Tomjg14/computing-science-day-demo/tree/v1-oude-demo).

---

## ✉️ Contact

Vragen? Mail naar [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl).
