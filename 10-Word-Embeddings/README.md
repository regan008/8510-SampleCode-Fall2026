# Word Embeddings

Train word2vec models on the APER corpus (American physical education journals, 1896–1959),
explore what they learned, then align models from different periods to see how words
changed meaning.

| Script | What it does | What you change |
|---|---|---|
| `01_train_model.py` | Trains a model on one folder of text | `CORPUS_FOLDER`, `MODEL_NAME` |
| `02_explore_model.py` | Nearest words, nearest to several words, a 2D plot of a word and its neighbors | `MODEL_NAME`, `WORD`, `SEED_WORDS`, `PLOT_COUNT` |
| `03_align_models.py` | Aligns the period models so they can be compared | (nothing, unless you rename models) |
| `04_explore_change.py` | Traces one word across the periods | `WORD` |
| `05_gender_and_a_word.py` | Which words near a target word lean toward women, and which toward men (after Schmidt) | `MODEL_NAME`, `TARGET`, `WOMEN_WORDS`, `MEN_WORDS` |
| `06_gender_over_time.py` | Where a few words fall between the women and men words in each period, on one ruler per period | `WORDS`, `WOMEN_WORDS`, `MEN_WORDS`, `EXTRA_RUNS` |

---

## Setup

### Step 1: Install Python 3.13

This week needs **Python 3.13**. The word2vec library we use (gensim) doesn't work with
Python 3.14 yet. You can keep 3.14 installed; 3.13 sits alongside it.

Check whether you already have it:

```bash
python3.13 --version            # Mac
py -3.13 --version              # Windows
```

If you get "command not found" (Mac) or "No suitable Python runtime found" (Windows), install it:

- **Mac with Homebrew:** `brew install python@3.13`
- **Mac without Homebrew, or Windows:** download the **Python 3.13** installer (not 3.14) from
  <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** on the
  first screen of the installer.

### Step 2: Create the virtual environment with Python 3.13

From inside the `10-Word-Embeddings` folder:

```bash
python3.13 -m venv venv         # Windows: py -3.13 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python --version                # should say Python 3.13.x
pip install -r requirements.txt
```

### Step 3: Download and unzip the corpus

The corpus is a 49 MB zip file on Box. Run these commands from inside the
`10-Word-Embeddings` folder.

**Mac (Terminal):**

```bash
curl -L -o aper-clean.zip "https://clemson.box.com/shared/static/ff0f2hzicbonuj1a7sczn3titp49ol8r.zip"
unzip -q aper-clean.zip -x "__MACOSX/*"
```

**Windows (PowerShell):**

```powershell
Invoke-WebRequest -Uri "https://clemson.box.com/shared/static/ff0f2hzicbonuj1a7sczn3titp49ol8r.zip" -OutFile aper-clean.zip
Expand-Archive aper-clean.zip -DestinationPath .
Remove-Item -Recurse -Force __MACOSX
```

The last command on each removes a folder of hidden Mac files that comes with the zip.

You should now have an **`aper-clean`** folder with one folder per period:

```
10-Word-Embeddings/
└── aper-clean/
    ├── 1896-1919/
    ├── 1920-1939/
    └── 1940-1959/
```

Check with `ls aper-clean` (Mac) or `dir aper-clean` (Windows): you should see the three period folders.
If the download doesn't work, open <https://clemson.box.com/s/ff0f2hzicbonuj1a7sczn3titp49ol8r> in a
browser, download the zip, move it into `10-Word-Embeddings`, and run just the unzip commands above.

---

## Running it

```bash
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**1. Train three models,** one per period. Run `01_train_model.py` three times, changing these
two lines at the top each time:

```python
CORPUS_FOLDER = SCRIPT_DIR / "aper-clean" / "1896-1919"
MODEL_NAME = "1896-1919"
```

```bash
python 01_train_model.py
```

Each model is saved in `models/`. Training takes a minute or two per period.

**2. Explore a model:** `python 02_explore_model.py`

**3. Align the three models:** `python 03_align_models.py`

**4. Trace a word over time:** `python 04_explore_change.py`

**5. Gender and a word:** `python 05_gender_and_a_word.py` (uses a model from step 1; no alignment needed)

**6. Gender over time:** `python 06_gender_over_time.py` (uses the models from step 1; no alignment
needed, because each period's women–men ruler is built inside its own model). Set `EXTRA_RUNS = 3`
to retrain each period a few times and see how much the positions move from run to run.

Plots are saved in `plots/` and open automatically.

### A message you can ignore

While training, you may see this a few times:

```
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'
```

It's a known, harmless bug in gensim ([issue #3665](https://github.com/piskvorky/gensim/issues/3665)).
The model still trains correctly.

---

## Things to keep in mind

- **Results change a little every time you train,** even with the same settings. Training is
  split across your computer's processor cores, and the order they finish in varies.
- **OCR errors become "words."** Fragments like `effi` or `recrea` show up as neighbors because the
  OCR split words apart. That's what the model learned from the text it was given.
- **The later journals have more OCR problems** (multi-column pages read straight across), so part
  of the difference between periods comes from the scanning, not the language.
