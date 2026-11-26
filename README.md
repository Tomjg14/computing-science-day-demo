# Kun jij de AI foppen? 🕶️

**🇳🇱 Nederlands** · [🇬🇧 English](README.en.md) · [🌐 Website](https://tomjg14.github.io/computing-science-day-demo/)

Een interactieve show over computer vision voor scholieren, gemaakt voor de Computing Science Day van de Radboud Universiteit.
Een webcam en een groot scherm laten in vier stappen zien hoe een computer naar beelden kijkt, hoe zeker een AI is, en hoe je een AI kunt foppen.

> ✉️ **Vragen, of loop je vast bij de installatie?** Mail me gerust: [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl)

> 🔬 **Het onderzoek achter de demo:** onderzoekers maakten in 2016 een bril met een speciaal patroon die een gezichtsherkenner liet denken dat de drager iemand anders was.
> Lees het artikel: [Sharif et al. (2016), *Accessorize to a Crime*](https://doi.org/10.1145/2976749.2978392). Meer in [Het onderzoek](#-het-onderzoek).

---

## 🎬 Wat zit er in de show?

| Stap | Wat zie je? | Wat leer je? |
|---|---|---|
| **Wachtscherm** | Uitnodiging en het scorebord van vandaag | – |
| **1. Wat ziet de computer?** | Een foto van je gezicht wordt pixels, en elke pixel wordt drie getallen (rood, groen, blauw). Met **E**: hoe een filter randen vindt, live (uitgebreid: een echt filter van het model op de pixels van de foto). | Voor een computer is een foto alleen getallen. Filters maken er aanwijzingen van, zoals randen. |
| **2. Hoe zeker is de computer?** | Gezichten en voorwerpen met een zekerheidsscore. Met vier spellen: *Bedrieg de AI!*, *Wat denkt de AI?*, *Speurtocht* en *Verwarringsmeter*. | Een AI geeft een score, geen zekerheid, en raakt snel in de war. |
| **3. Verstop je voor de AI!** | Spel: de AI ziet je lichaam, maar niet je gezicht = onzichtbaar. Makkelijk (oude AI uit 2001) of moeilijk (moderne AI). Eerst met **E**: waarom de AI een gezicht kan missen, met de live gezichtsscore en metingen van je gezicht. | Waarom een AI een gezicht mist, en waarom de oude AI makkelijker te foppen is dan de moderne. |
| **4. Zo doen onderzoekers het** | Een *simulatie* van de bril uit het onderzoek, met uitleg hoe zo'n bril gemaakt wordt. | Kleine veranderingen op de juiste plek kunnen een AI foppen. |

De show draait helemaal op je eigen computer. Hij slaat geen foto's of video's op (zie [Privacy](#-privacy)).

---

## 🎓 Groepsles (20–30 leerlingen)

Naast de show voor voorbijgangers heeft de demo een tweede variant: een les van ongeveer 20–25 minuten voor één groep.
De groep staat in een halve cirkel voor het scherm. Een paar vrijwilligers staan vooraan en spelen mee; de hele groep speelt het groepsspel.

```bash
python -m show --lesson
```

Of druk tijdens de show op **G** (of op de knop in het paneel) om te wisselen. De show voor voorbijgangers verandert hier niet door.

| Hoofdstuk | Wat zie je? |
|---|---|
| **1. Opening** | De AI telt de groep: "Ik zie ongeveer 27 mensen". Iedereen krijgt een dun, licht kader; de spelers vooraan een gewoon kader. |
| **2. Pixels** | Stap 1 van de show. Daarna **E**: hoe een filter randen vindt (uitgebreid: één echt filter van het model op de pixels van de foto). |
| **3. Hoe herkent de AI dingen?** | Drie uitlegpagina's, live met echte getallen: de oude AI (2001), de moderne AI (lagen met filters), en hoe de aanwijzingen een kader worden. **E** gaat naar de volgende pagina; na de laatste sluit **E** de uitleg. **Page Down** toont de tekst stap voor stap. |
| **4. Hoe zeker is de computer?** | Stap 2 van de show, met de vier spellen. *Wat denkt de AI?* werkt goed met de hele groep. |
| **5. Verstop je voor de AI!** | Eerst de pagina *Waarom kan de AI een gezicht missen?* (**E** sluit hem), dan stap 3 van de show, met de vrijwilligers vooraan. |
| **6. Raak het getal!** | Het groepsspel (zie hieronder). |
| **7. De bril** | Stap 4 van de show (een *simulatie*). |
| **8. Verkeersbord** | Een eigen tekening van een stopbord met stickers: hetzelfde idee in een ander voorbeeld. Een *illustratie*: geen model kijkt ernaar. Bron: Eykholt et al. (2018). |
| **9. Afsluiting** | Bedankt, en een grote QR-code. |

**Raak het getal!** De AI moet precies een doelgetal aan gezichten tellen. De groep beslist wie zijn gezicht verstopt.

1. Iedereen kijkt in de camera. Elk gezicht dat de AI ziet, krijgt een stip. **SPATIE** meet het maximum. Gezichten die de camera niet ziet, tellen niet mee, dus elk doel is haalbaar.
2. Het doel begint op het maximum min 1. Pas het aan met **↑ / ↓** of in het paneel.
3. De groep overlegt. Klaar? **SPATIE** start het aftellen. Dan telt de AI 2 seconden lang; het middelste getal telt, zodat één flikkering niet beslist.
4. Het beeld bevriest, met een stip op elk getelde gezicht en de uitslag. **SPATIE**: volgende ronde. **BACKSPACE**: maximum opnieuw meten.

Het spel bewaart geen scores. De les heeft geen wachtscherm: hij begint bij de opening, en na de afsluiting gaat hij terug naar de opening. Is er even niemand in beeld, dan blijft de les gewoon staan.
Het aantal spelers vooraan (standaard 4) stel je in het paneel in bij *Max players*.

**Testen zonder groep.** `python -m show --lesson --test` (of **Y**) voegt een gesimuleerde groep toe aan de resultaten van de AI: elke persoon heeft een lichaam en een gezicht. **,** verstopt één gezicht, **.** laat er weer één zien; het lichaam blijft, net als in het echte spel. **Shift+,** / **Shift+.** haalt een hele persoon weg of voegt er een toe. Zolang de testmodus aan staat, staat er een rood label op het scherm. De begingrootte van de groep staat in `show/config.py` (`TEST_FACES`).

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

## 👩‍🏫 Voor de docent of presentator

Dit deel is voor wie de demo geeft. Het vertelt welke versie je kiest, wat je voorbereidt en hoe je de uitleg leidt.

### Twee versies

| | **Show voor voorbijgangers** | **Groepsles** |
|---|---|---|
| Voor | Een stand op een open dag: mensen lopen langs, kijken en spelen mee | Eén klas van 20–30 leerlingen, ongeveer 20–25 minuten |
| Starten | `python -m show` | `python -m show --lesson` |
| Verloop | Wachtscherm → **SPATIE** → stap 1–4 | Hoofdstuk 1–9, van de opening tot de afsluiting |
| Spelers | Tot 6 spelers in beeld | Een paar vrijwilligers vooraan (standaard 4), plus groepsspellen voor iedereen |
| 30 s niemand in beeld | Terug naar het wachtscherm | Blijft waar hij is |

Druk op **G** (of op de knop in het paneel) om tijdens de show te wisselen tussen de twee versies.

### Twee niveaus van uitleg

Alle uitlegpagina's (toets **E**) hebben twee niveaus:

- **Eenvoudig** (standaard): één idee per pagina, in gewone taal, voor leerlingen zonder technische achtergrond.
- **Uitgebreid**: meer stappen en de echte getallen (gewichten, testwaarden, de 5 hoogste scores). Goed voor oudere leerlingen of een technisch publiek.

**Shift+E** (of *Level* in het paneel) wisselt alle uitleg tegelijk. Het verhaal is op beide niveaus hetzelfde:

1. Voor een computer is een foto alleen getallen (pixels).
2. Filters maken van die getallen aanwijzingen, zoals randen.
3. De oude AI (2001) stelt vaste ja/nee-vragen en stopt bij het eerste NEE. De moderne AI heeft zijn filters zelf geleerd, telt veel aanwijzingen op en vindt alle voorwerpen in één keer kijken (YOLO).
4. De AI kan een gezicht missen als hij te weinig aanwijzingen krijgt.
5. Met kleine veranderingen op de juiste plek kun je een AI foppen. Onderzoekers doen dit om AI sterker en veiliger te maken.

### Voordat je begint

- **Camera en licht:** het licht valt op de gezichten, niet van achteren (geen raam achter de groep). Controleer het beeld met **C** (volgende camera).
- **Schermen:** de show op het grote scherm, het bedieningspaneel op je laptop (zie [Opstelling](#️-opstelling-thuis-of-in-de-klas)).
- **Taal:** **L** wisselt het scherm tussen Nederlands en Engels.
- **Clicker:** de meeste presentatie-clickers sturen *Page Down*, en dat toont de volgende tekststap. Test dit eerst: sommige clickers sturen de pijltjestoetsen, en die wisselen de stap of het hoofdstuk.
- **Groepsles:** zet *Max players* in het paneel, en plak een vak op de vloer voor de vrijwilligers.
- **Alleen oefenen:** voeg `--test` toe (of druk op **Y**) voor een gesimuleerde groep. Zie [Testen zonder groep](#-groepsles-2030-leerlingen).

### De uitleg leiden

- **E** opent een uitlegpagina. Een pagina opent met alleen de titel en het live beeld. De tekst wacht op jou.
- **Page Down** (of **N**, of de clicker) toont de volgende tekststap; **Page Up** gaat één stap terug. Niets gaat vanzelf verder, dus je kunt wachten op een goed beeld, op aandacht of op een vraag. De bolletjes onderaan tonen hoeveel stappen er zijn.
- De pagina's volgen één gezicht: het dichtstbijzijnde, één keer gekozen. Ze wisselen nooit vanzelf naar een ander gezicht. **TAB** kiest het volgende gezicht, bijvoorbeeld als er een andere vrijwilliger komt.
- Alles op de uitlegpagina's is live en komt uit de echte modellen. Delen die alleen een illustratie zijn (het vertraagde zoeken van de oude AI, het patroon van de bril) hebben een label op het scherm.

### Groepsles: hoofdstuk voor hoofdstuk

| # | Hoofdstuk | Wat doe je? |
|---|---|---|
| 1 | Opening: "Ik zie ongeveer N mensen" | Laat de groep binnenkomen en naar de telling kijken. |
| 2 | Pixels | **SPATIE**: foto van een vrijwilliger; **BACKSPACE**: weer live beeld. **E**: hoe een filter randen vindt (live: laat de groep zwaaien). |
| 3 | Hoe herkent de AI dingen? | De pagina over de oude AI opent vanzelf. **E**: moderne AI → van aanwijzingen naar een kader → sluiten (daarna vrije detectie). |
| 4 | Hoe zeker is de computer? | **1** Bedrieg, **2** Raad (hele groep), **3** Speurtocht, **4** Verwarring, **0** vrij. **I**: uitlegkaart van het spel. |
| 5 | Waarom de AI mist + Verstop je voor de AI! | De pagina *Waarom kan de AI een gezicht missen?* opent vanzelf: laat een vrijwilliger het gezicht bedekken, draaien en naar achteren stappen. **E** sluit hem. **SPATIE**: start een ronde. **M**: makkelijk / moeilijk. |
| 6 | Raak het getal! | **SPATIE**: meet het maximum. **↑ / ↓**: doel. **SPATIE**: aftellen. **SPATIE**: volgende ronde. |
| 7 | De bril (simulatie) | **E**: hoe de bril gemaakt wordt, en waarom onderzoekers dit doen. |
| 8 | Verkeersbord (illustratie) | Hetzelfde idee in een ander voorbeeld. |
| 9 | Afsluiting | De QR-code naar de website (**K**: aan / uit). |

### Als er iets misgaat

- **Een vrijwilliger wordt geen speler:** druk op **D** (debug). Het paneel toont per persoon de reden. Vaak helpt een stap dichterbij, iets uit elkaar gaan staan, of recht in de camera kijken. Zet debug weer uit: de tekst staat ook op het grote scherm.
- **De uitleg volgt het verkeerde gezicht:** druk op **TAB** tot het goede gezicht gekozen is.
- **Stap 3 werkt niet in deze ruimte (licht, afstand):** druk op **P** om naar stap 4 te springen.
- Meer problemen en oplossingen: zie [Problemen oplossen](#-problemen-oplossen).

---

## 🎮 Tijdens de show

| Toets | Wat doet het? |
|---|---|
| **SPATIE** | Start de sessie (wachtscherm) of de actie van de stap: foto maken, ronde starten. Tijdens een ronde: ronde nu stoppen en de score tonen (tijdens het aftellen: ronde afbreken). |
| **X** | Tijdens een ronde (spel in stap 2, stap 3): 10 seconden extra. Nog eens drukken geeft meer tijd. |
| **→ / ←** | Volgende / vorige stap. |
| **G** | Wissel tussen de show voor voorbijgangers en de groepsles. |
| **Y** | Testmodus aan / uit: een gesimuleerde groep, om de les zonder groep te testen. **,** / **.**: één gezicht verstoppen / weer laten zien. Met **Shift**: één persoon weghalen / toevoegen. |
| **↑ / ↓** | *Raak het getal!*: doel één hoger / lager. |
| **BACKSPACE** | Stap 1: terug naar het live beeld. *Raak het getal!*: maximum opnieuw meten. |
| **ESC** | Terug naar het wachtscherm (groepsles: de opening). Staat er een scorebord op het scherm, dan sluit ESC alleen dat. |
| **1 – 4 / 0** | Stap 2: kies een spel / terug naar vrij kijken. |
| **I** | Uitleg "hoe speel je het" aan / uit. |
| **Shift+E** | Alle uitleg: eenvoudig (standaard) / uitgebreid. |
| **Page Down** / **N** | Uitleg open: toon de volgende tekststap (een presentatie-clicker stuurt Page Down). Een uitlegpagina opent met alleen de titel en het beeld; niets gaat vanzelf verder. |
| **Page Up** | Uitleg open: één tekststap terug. |
| **TAB** | De uitleg volgt een ander gezicht. De uitleg kiest één keer het dichtstbijzijnde gezicht en wisselt nooit vanzelf. |
| **E** | Uitleg, live met echte getallen. Standaard eenvoudig (voor iedereen); **Shift+E** zet alle uitleg op uitgebreid. Stap 1: hoe een filter randen vindt (uitgebreid: na de foto, één filter op je pixels). Stap 2: oude AI → moderne AI → van aanwijzingen naar een kader → uit. Stap 3: waarom de AI een gezicht kan missen. Stap 4: hoe de bril gemaakt wordt. |
| **M** | Stap 3: makkelijk (oude AI) / moeilijk (moderne AI). |
| **V** | Scores op gezichten aan / uit (stap 3: alleen tijdens een ronde). |
| **T** | Stap 2: volgend doel (*Bedrieg de AI!*, Shift+T: vorig doel) of kies één voorwerp (*Verwarringsmeter*; na het laatste nummer weer automatisch: alle vastgehouden voorwerpen tellen; je kunt ook een voorwerp aanklikken in het paneel). |
| **O** | Overlays aan / uit: titelbalk, infobalken en QR-code. Spelmeters en tijd blijven staan. |
| **K** | QR-code aan / uit. |
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
- De show slaat **geen foto's of video's** op. De foto in stap 1 staat alleen in het geheugen en verdwijnt als je verdergaat. Hetzelfde geldt voor het bevroren beeld van *Raak het getal!*: het verdwijnt bij de volgende ronde.
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
│   ├── players.py         wie speelt er mee, en waarom iemand (nog) geen speler is
│   ├── testmode.py        een gesimuleerde groep, om zonder groep te testen (toets Y)
│   ├── stages/            de stappen: wachtscherm, 1 pixels, 2 detectie, 3 verstoppen, 4 bril,
│   │                      en de hoofdstukken van de groepsles (lesson.py, hitnumber.py)
│   └── games/             de vier spellen van stap 2
├── assets/                plaatjes: de bril en de QR-code
├── docs/                  de website (GitHub Pages)
├── tests/                 tests (spellogica; de filterpagina rekent als het model): python -m unittest discover -s tests -t .
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
| Iemand vooraan wordt geen speler | Zet debug aan (**D**). Het paneel en het scherm tonen per persoon waarom die geen speler is. Vaak helpt: een stap dichterbij, iets uit elkaar gaan staan, recht in de camera kijken. Lukt het niet, loop dan even uit beeld en weer terug. |

---

## 🕰️ De vorige versie

De eerste versie van deze demo (met een Tkinter-venster en gezichtsherkenning) staat nog in de geschiedenis van het project, onder de tag [`v1-oude-demo`](https://github.com/Tomjg14/computing-science-day-demo/tree/v1-oude-demo).

---

## ✉️ Contact

Vragen? Mail naar [tom.janssen-groesbeek@ru.nl](mailto:tom.janssen-groesbeek@ru.nl).
