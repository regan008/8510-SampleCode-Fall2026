#!/usr/bin/env python3
"""
Step 2: Explore one model (following Schmidt)
=============================================

Three ways of looking inside a trained model:

  1. Nearest words: the words whose vectors are closest to one word,
     with their similarity scores.
     (Schmidt: nearest_to(chronam_vectors[["funny"]]))

  2. Nearest to several words at once: average a few seed words into one
     vector and find the words closest to it. A way to map out a whole
     semantic field. (Schmidt: nearest_to(...[[c("oysters","ham","bread","chicken")]], 20))

  3. A "non-optimal positioning of words in space": flatten the 100
     dimensions down to 2 and plot WORD with its nearest neighbors. Some
     groupings make sense; others don't, because 2 dimensions can't hold
     all the relationships in the model. (After Schmidt's 2D plot)

Change the settings below and run it again.
"""

import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, not a window
import matplotlib.pyplot as plt
from adjustText import adjust_text
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent

# --- Settings ------------------------------------------------------------------

MODEL_NAME = "1940-1959"              # which model in models/ to explore

WORD = "fitness"                      # 1. nearest words to this word (also used in 3.)
TOP_N = 15

SEED_WORDS = ["basketball", "baseball", "football", "tennis"]   # 2. a semantic field
FIELD_SIZE = 20

PLOT_COUNT = 40                       # 3. how many of WORD's nearest neighbors to plot

OPEN_PLOTS = True                     # open the plot automatically when it's saved

# -------------------------------------------------------------------------------

MODEL_FOLDER = SCRIPT_DIR / "models"
PLOT_FOLDER = SCRIPT_DIR / "plots"


def nearest_words(model, word, n):
    """1. The n words closest to one word."""
    print(f"\n1. Nearest words to '{word}'  (1.0 = identical direction, 0 = unrelated)")
    if word not in model.wv:
        print(f"   '{word}' isn't in this model's vocabulary (too rare, or misspelled).")
        return
    for neighbor, score in model.wv.most_similar(word, topn=n):
        print(f"   {neighbor:22} {score:.3f}")


def nearest_to_several(model, seeds, n):
    """2. Average several words into one vector and find what's near it."""
    found = [w for w in seeds if w in model.wv]
    missing = [w for w in seeds if w not in model.wv]
    print(f"\n2. Words nearest to the average of {found}")
    if missing:
        print(f"   (not in the vocabulary, skipped: {missing})")
    if not found:
        return
    neighbors = [w for w, _ in model.wv.most_similar(positive=found, topn=n)]
    print("   " + ", ".join(neighbors))


def plot_positioning(model, word, count):
    """3. Plot `word` and its `count` nearest neighbors in 2 dimensions."""
    if word not in model.wv:
        print(f"\n3. Can't plot: '{word}' isn't in this model's vocabulary.")
        return
    words = [word] + [w for w, _ in model.wv.most_similar(word, topn=count)]
    vectors = model.wv[words]

    # PCA finds the 2 directions that spread these words out the most
    points = PCA(n_components=2).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.scatter(points[1:, 0], points[1:, 1], s=8, color="#8a93a6")
    ax.scatter(points[0, 0], points[0, 1], s=60, color="#c0392b", zorder=3)   # the word itself
    texts = [ax.text(x, y, neighbor, fontsize=11) for (x, y), neighbor in zip(points[1:], words[1:])]
    texts.append(ax.text(points[0, 0], points[0, 1], word, fontsize=15, fontweight="bold",
                         color="#c0392b", zorder=4))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"'{word}' and its {count} nearest neighbors, "
                 f"flattened to 2 dimensions ({MODEL_NAME})")
    ax.spines[["top", "right"]].set_visible(False)

    # Nudge the labels apart so they don't overlap; a thin line joins a moved label to its dot
    adjust_text(texts, x=points[:, 0], y=points[:, 1], ax=ax,
                arrowprops=dict(arrowstyle="-", color="#8a93a6", lw=0.6))

    PLOT_FOLDER.mkdir(exist_ok=True)
    path = PLOT_FOLDER / f"neighbors_{word}_{MODEL_NAME}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n3. Saved the plot to {path.relative_to(SCRIPT_DIR)}")
    if OPEN_PLOTS:
        webbrowser.open(path.resolve().as_uri())


def main():
    path = MODEL_FOLDER / f"{MODEL_NAME}.model"
    if not path.exists():
        print(f"Can't find {path.relative_to(SCRIPT_DIR)}. Train it first with 01_train_model.py.")
        return
    model = Word2Vec.load(str(path))
    print(f"Model: {MODEL_NAME}  ({len(model.wv):,} words)")

    nearest_words(model, WORD, TOP_N)
    nearest_to_several(model, SEED_WORDS, FIELD_SIZE)
    plot_positioning(model, WORD, PLOT_COUNT)


if __name__ == "__main__":
    main()
