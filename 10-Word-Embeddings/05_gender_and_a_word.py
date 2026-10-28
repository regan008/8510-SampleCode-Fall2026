#!/usr/bin/env python3
"""
Step 5: Which words near a target word lean toward women, and which toward men?
===============================================================================

Following Schmidt, "Rejecting the gender binary: a vector-space operation" (2015).

Every word in the model gets two scores:

  1. How close it is to TARGET (e.g. "fitness"): cosine similarity.
  2. Which way it leans on a women <-> men direction. The direction is the
     average of WOMEN_WORDS minus the average of MEN_WORDS, so a positive
     score leans toward the women words and a negative score toward the men words.

Then we take the words closest to TARGET and sort them by the second score.
That answers: "Of the words associated with fitness, which are used more
like women-words, and which more like men-words?"

Uses a model you already trained with 01_train_model.py. Change the settings
and run it again; the two word lists can be any two groups you want to contrast.
"""

import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, not a window
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from gensim.models import Word2Vec

SCRIPT_DIR = Path(__file__).parent

# --- Settings ------------------------------------------------------------------

MODEL_NAME = "1896-1919"              # which model in models/ to use
TARGET = "fitness"

WOMEN_WORDS = ["woman", "women", "she", "her", "hers", "herself", "girl", "girls"]
MEN_WORDS = ["man", "men", "he", "his", "him", "himself", "boy", "boys"]

NEAR_SHARE = 0.03       # look at the 3% of words closest to TARGET
MIN_COUNT = 50          # skip rarer words (cuts out most OCR fragments)
SHOW = 20               # how many words to list on each side
PLOT_LABELS = 15        # how many of those to label on the plot

OPEN_PLOTS = True

# -------------------------------------------------------------------------------

MODEL_FOLDER = SCRIPT_DIR / "models"
PLOT_FOLDER = SCRIPT_DIR / "plots"


def unit(vector):
    """Scale a vector to length 1, so only its direction matters."""
    return vector / np.linalg.norm(vector)


def main():
    path = MODEL_FOLDER / f"{MODEL_NAME}.model"
    if not path.exists():
        print(f"Can't find {path.relative_to(SCRIPT_DIR)}. Train it first with 01_train_model.py.")
        return
    wv = Word2Vec.load(str(path)).wv
    if TARGET not in wv:
        print(f"'{TARGET}' isn't in the {MODEL_NAME} vocabulary.")
        return

    women = [w for w in WOMEN_WORDS if w in wv]
    men = [w for w in MEN_WORDS if w in wv]
    print(f"Model: {MODEL_NAME}")
    print(f"Women words: {', '.join(women)}")
    print(f"Men words:   {', '.join(men)}")

    # The gender direction: average of the women words minus average of the men words
    gender_direction = unit(np.mean(wv[women], axis=0) - np.mean(wv[men], axis=0))
    target = unit(wv[TARGET])

    # Two scores for every common word
    words = [w for w in wv.index_to_key
             if wv.get_vecattr(w, "count") >= MIN_COUNT and w not in women + men + [TARGET]]
    vectors = np.array([unit(wv[w]) for w in words])
    closeness = vectors @ target              # 1. how close to TARGET
    gender = vectors @ gender_direction       # 2. + leans women, - leans men

    # Keep only the words closest to TARGET, then sort them by gender score
    cutoff = np.quantile(closeness, 1 - NEAR_SHARE)
    near = [i for i in range(len(words)) if closeness[i] >= cutoff]
    lean_women = sorted(near, key=lambda i: gender[i], reverse=True)[:SHOW]
    lean_men = sorted(near, key=lambda i: gender[i])[:SHOW]

    print(f"\nThe {len(near)} words closest to '{TARGET}' (similarity {cutoff:.2f} or more), "
          f"sorted by which way they lean:")
    print(f"\n  Lean toward WOMEN:")
    print("   " + ", ".join(f"{words[i]} ({gender[i]:+.2f})" for i in lean_women))
    print(f"\n  Lean toward MEN:")
    print("   " + ", ".join(f"{words[i]} ({gender[i]:+.2f})" for i in lean_men))
    print(f"\n  ('{TARGET}' itself scores {float(target @ gender_direction):+.2f})")

    # Plot: gender score across, closeness to TARGET up the side
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.scatter(gender[near], closeness[near], s=10, color="#c5cad3")
    texts = [ax.text(gender[i], closeness[i], words[i], fontsize=12, color="#8e44ad")
             for i in lean_women[:PLOT_LABELS]]
    texts += [ax.text(gender[i], closeness[i], words[i], fontsize=12, color="#2a6f97")
              for i in lean_men[:PLOT_LABELS]]
    ax.axvline(0, color="#8a93a6", lw=1, ls="--")
    ax.set_xlabel("← leans toward men words          leans toward women words →", fontsize=13)
    ax.set_ylabel(f"similarity to '{TARGET}'", fontsize=13)
    ax.set_title(f"Words closest to '{TARGET}', by gender lean ({MODEL_NAME})", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)

    # Nudge the labels apart so they don't overlap; a thin line joins a moved label to its dot
    adjust_text(texts, x=gender[near], y=closeness[near], ax=ax,
                arrowprops=dict(arrowstyle="-", color="#8a93a6", lw=0.6))

    PLOT_FOLDER.mkdir(exist_ok=True)
    plot_path = PLOT_FOLDER / f"gender_{TARGET}_{MODEL_NAME}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved the plot to {plot_path.relative_to(SCRIPT_DIR)}")
    if OPEN_PLOTS:
        webbrowser.open(plot_path.resolve().as_uri())


if __name__ == "__main__":
    main()
