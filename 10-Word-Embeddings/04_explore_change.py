#!/usr/bin/env python3
"""
Step 4: How does a word change over time?
=========================================

Uses the aligned models from 03_align_models.py to trace one word across
the periods:

  1. How often it appears in each period
  2. Its nearest neighbors in each period
  3. How far it moved between periods, compared with a typical word
  4. A plot of its path through the space, with its neighbors in each
     period (like Hamilton et al.'s figure of "gay", "broadcast", "awful")

Change WORD and run it again.
"""

import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, not a window
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent

# --- Settings ------------------------------------------------------------------

WORD = "fitness"

# The same names, in the same order, as in 03_align_models.py
MODEL_NAMES = ["1896-1919", "1920-1939", "1940-1959"]

TOP_N = 10                  # neighbors to list for each period
PLOT_NEIGHBORS = 6          # neighbors to draw for each period
COMPARE_MIN_COUNT = 200     # compare against words used this often in both periods

OPEN_PLOTS = True

# -------------------------------------------------------------------------------

ALIGNED_FOLDER = SCRIPT_DIR / "models" / "aligned"
PLOT_FOLDER = SCRIPT_DIR / "plots"
COLORS = ["#2a6f97", "#c0392b", "#2f7d6b", "#8e44ad", "#d35400"]


def cosine_distance(a, b):
    """0 = pointing the same way (no change); higher = more change.
    The aligned vectors all have length 1, so this is 1 - their dot product."""
    return 1 - float(a @ b)


def compare_to_other_words(first, last, distance):
    """Every word moves a little (the models are trained separately, and the
    OCR changes over time). So compare: what share of common words moved
    less than this one?"""
    common = [w for w in first.index_to_key if w in last
              and first.get_vecattr(w, "count") >= COMPARE_MIN_COUNT
              and last.get_vecattr(w, "count") >= COMPARE_MIN_COUNT]
    distances = np.array([cosine_distance(first[w], last[w]) for w in common])
    return float(np.mean(distances < distance) * 100), len(common)


def plot_path(models, word):
    """Draw the word's position in each period, joined by arrows, with its
    nearest neighbors from each period in gray."""
    names = [n for n in MODEL_NAMES if word in models[n]]
    labels, vectors, kinds = [], [], []
    for name in names:
        labels.append(f"{word} ({name})"); vectors.append(models[name][word]); kinds.append(name)
    seen = set()
    for name in names:
        for neighbor, _ in models[name].most_similar(word, topn=PLOT_NEIGHBORS):
            if neighbor not in seen:
                seen.add(neighbor)
                labels.append(neighbor); vectors.append(models[name][neighbor]); kinds.append("neighbor")

    points = PCA(n_components=2).fit_transform(np.array(vectors))

    fig, ax = plt.subplots(figsize=(12, 9))
    texts = []
    neighbor_points = points[len(names):]
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], s=8, color="#c5cad3")
    for (x, y), label, kind in zip(points, labels, kinds):
        if kind == "neighbor":
            texts.append(ax.text(x, y, label, fontsize=12, color="#8a93a6"))
    word_points = points[:len(names)]
    for i, (name, (x, y)) in enumerate(zip(names, word_points)):
        color = COLORS[i % len(COLORS)]
        ax.scatter(x, y, s=60, color=color, zorder=3)
        texts.append(ax.text(x, y, f"{word} ({name})", fontsize=14, fontweight="bold",
                             color=color, zorder=4))
        if i:
            ax.annotate("", xy=(x, y), xytext=tuple(word_points[i - 1]),
                        arrowprops=dict(arrowstyle="->", color="#4a5568", lw=2), zorder=2)
    pad = 0.15 * (points.max(axis=0) - points.min(axis=0))
    ax.set_xlim(points[:, 0].min() - pad[0], points[:, 0].max() + pad[0])
    ax.set_ylim(points[:, 1].min() - pad[1], points[:, 1].max() + pad[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"How '{word}' moved, {names[0]} to {names[-1]}")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # Nudge the labels apart so they don't overlap; a thin line joins a moved label to its dot
    adjust_text(texts, x=points[:, 0], y=points[:, 1], ax=ax,
                arrowprops=dict(arrowstyle="-", color="#8a93a6", lw=0.6))

    PLOT_FOLDER.mkdir(exist_ok=True)
    path = PLOT_FOLDER / f"change_{word}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n4. Saved the plot to {path.relative_to(SCRIPT_DIR)}")
    if OPEN_PLOTS:
        webbrowser.open(path.resolve().as_uri())


def main():
    models = {}
    for name in MODEL_NAMES:
        path = ALIGNED_FOLDER / f"{name}.kv"
        if not path.exists():
            print(f"Can't find {path.relative_to(SCRIPT_DIR)}. Run 03_align_models.py first.")
            return
        models[name] = KeyedVectors.load(str(path))

    missing = [n for n in MODEL_NAMES if WORD not in models[n]]
    if missing:
        print(f"'{WORD}' isn't in the vocabulary for: {', '.join(missing)} (too rare there).")
    present = [n for n in MODEL_NAMES if WORD in models[n]]
    if len(present) < 2:
        print("It needs to appear in at least two periods to compare.")
        return

    print(f"1. How often '{WORD}' appears")
    for name in present:
        print(f"   {name}: {models[name].get_vecattr(WORD, 'count'):,} times")

    print(f"\n2. Nearest neighbors in each period")
    for name in present:
        neighbors = ", ".join(w for w, _ in models[name].most_similar(WORD, topn=TOP_N))
        print(f"   {name}: {neighbors}")

    print(f"\n3. How far '{WORD}' moved  (0 = no change)")
    for earlier, later in zip(present, present[1:]):
        d = cosine_distance(models[earlier][WORD], models[later][WORD])
        print(f"   {earlier} -> {later}: {d:.2f}")
    first, last = present[0], present[-1]
    d = cosine_distance(models[first][WORD], models[last][WORD])
    share, n = compare_to_other_words(models[first], models[last], d)
    print(f"   {first} -> {last}: {d:.2f}")
    print(f"   '{WORD}' moved more than {share:.0f}% of the {n:,} words used "
          f"{COMPARE_MIN_COUNT}+ times in both periods")

    plot_path(models, WORD)


if __name__ == "__main__":
    main()
