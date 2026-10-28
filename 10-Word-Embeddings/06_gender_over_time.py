#!/usr/bin/env python3
"""
Step 6: Does a word lean toward women or men differently over time?
===================================================================

Combines 04 (change over time) with 05 (gender and a word). For each
period's model, it builds 05's women <-> men direction INSIDE that model
and measures where each of WORDS falls on it. Every period gets its own
ruler, so the scores can be compared across periods without aligning the
models (you don't need 03_align_models.py for this one).

The plot draws one horizontal ruler per period:
  - the gray diamonds are its two ends: where the women words and the men
    words sit on average
  - the big dot is where each word sits
  - the gray shape shows where every other common word falls, for scale

Every training run comes out a little different. Set EXTRA_RUNS to train
each period a few more times: each run then shows as a small dot, and a
change between periods only counts if it's bigger than the spread between
runs.

Uses the models you trained with 01_train_model.py. As in 05, the two word
lists can be any two groups you want to contrast.
"""

import os
import re
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, not a window
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from gensim.models import Word2Vec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).parent

# --- Settings ------------------------------------------------------------------

WORDS = ["fitness", "athletics"]      # one to three words to place on the rulers

# The same names, in the same order, as the models you trained with 01_train_model.py
MODEL_NAMES = ["1896-1919", "1920-1939", "1940-1959"]

WOMEN_WORDS = ["woman", "women", "she", "her", "hers", "herself", "girl", "girls"]
MEN_WORDS = ["man", "men", "he", "his", "him", "himself", "boy", "boys"]

MIN_COUNT = 50          # words used this often make up the gray background shape

EXTRA_RUNS = 0          # extra models to train per period (try 3; each takes as long as 01 did)
CORPUS_FOLDER = SCRIPT_DIR / "aper-clean"   # extra runs read the text in CORPUS_FOLDER/<model name>/

OPEN_PLOTS = True

# -------------------------------------------------------------------------------

MODEL_FOLDER = SCRIPT_DIR / "models"
PLOT_FOLDER = SCRIPT_DIR / "plots"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
FILL, BASE = "#e9e8e2", "#c3c2b7"

# The same word pattern as 01_train_model.py
WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")


def read_corpus(folder):
    """Read and split the text exactly as 01_train_model.py does."""
    sentences = []
    for path in sorted(folder.rglob("*.txt")):
        for paragraph in path.read_text(encoding="utf-8").split("\n\n"):
            words = WORD.findall(paragraph.lower())
            for i in range(0, len(words), 1000):
                sentences.append(words[i:i + 1000])
    return sentences


def unit(vector):
    """Scale a vector to length 1, so only its direction matters."""
    return vector / np.linalg.norm(vector)


def ruler(wv):
    """05's women <-> men direction, built inside one model."""
    women = [w for w in WOMEN_WORDS if w in wv]
    men = [w for w in MEN_WORDS if w in wv]
    return unit(np.mean(wv[women], axis=0) - np.mean(wv[men], axis=0)), women, men


def place_words(wv):
    """Where each of WORDS falls on this model's ruler (+ women, - men; None if missing)."""
    direction, _, _ = ruler(wv)
    return {w: float(unit(wv[w]) @ direction) if w in wv else None for w in WORDS}


def measure(model, name):
    """Measure one period: its ruler's ends, the background words, and WORDS
    in the saved model plus any extra runs."""
    wv = model.wv
    direction, women, men = ruler(wv)
    common = [w for w in wv.index_to_key
              if wv.get_vecattr(w, "count") >= MIN_COUNT and w not in women + men]
    period = {
        "size": getattr(model, "corpus_total_words", 0),
        "counts": {w: wv.get_vecattr(w, "count") if w in wv else 0 for w in WORDS},
        "women_end": float(np.mean([unit(wv[w]) @ direction for w in women])),
        "men_end": float(np.mean([unit(wv[w]) @ direction for w in men])),
        "women_label": ", ".join(women[:3]) + " …",
        "men_label": ", ".join(men[:3]) + " …",
        "background": np.array([unit(wv[w]) for w in common]) @ direction,
        "runs": [place_words(wv)],
    }

    if EXTRA_RUNS:
        folder = CORPUS_FOLDER / name
        if not folder.exists():
            print(f"Can't find {folder} for the extra runs, so {name} uses the saved model only.")
            return period
        print(f"Retraining {name} {EXTRA_RUNS} more time{'s' if EXTRA_RUNS > 1 else ''} "
              f"with the saved model's settings ...", flush=True)
        sentences = read_corpus(folder)
        for seed in range(1, EXTRA_RUNS + 1):
            extra = Word2Vec(sentences=sentences, vector_size=model.vector_size, window=model.window,
                             min_count=model.min_count, sg=model.sg, epochs=model.epochs,
                             workers=os.cpu_count(), seed=seed)
            period["runs"].append(place_words(extra.wv))
    return period


def plot_rulers(periods):
    """One horizontal ruler per period, earliest at the top, with WORDS placed on each."""
    names = list(periods)
    height = 2.6 + 1.6 * (len(names) - 0.13)
    fig, ax = plt.subplots(figsize=(11, height), facecolor=SURFACE)
    fig.subplots_adjust(left=0.15, right=0.97, top=1 - 1.73 / height, bottom=0.86 / height)
    ax.set_facecolor(SURFACE)

    rows = {name: len(names) - 1 - i for i, name in enumerate(names)}
    everything = [s for p in periods.values() for run in p["runs"] for s in run.values() if s is not None]
    everything += [x for p in periods.values() for x in (p["women_end"], p["men_end"])]
    lo, hi = min(everything) - 0.12, max(everything) + 0.12
    xs = np.linspace(lo, hi, 400)

    means, texts, points = {}, [], []
    for name, y in rows.items():
        p = periods[name]
        # Every common word in this period, as a faint shape above the ruler
        density = gaussian_kde(p["background"])(xs)
        ax.fill_between(xs, y, y + 0.42 * density / density.max(), color=FILL, lw=0, zorder=1)
        ax.plot([lo, hi], [y, y], color=BASE, lw=1, zorder=2)

        # The ruler's two ends (labeled on the top ruler only)
        for end, label in ((p["women_end"], p["women_label"]), (p["men_end"], p["men_label"])):
            ax.scatter([end], [y], marker="D", s=34, color=MUTED, lw=0, zorder=3)
            points.append((end, y))
            if name == names[0]:
                texts.append(ax.text(end, y - 0.09, "average of\n" + label, ha="center", va="top",
                                     fontsize=9.5, color=MUTED, linespacing=1.2))

        # Each run as a small dot, the average of the runs as a big one
        ax.text(-0.02, y + 0.02, name, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(-0.02, y - 0.02, f"{p['size'] / 1e6:.1f}M words", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=10, color=MUTED)
        for word, color in zip(WORDS, COLORS):
            runs = [run[word] for run in p["runs"] if run[word] is not None]
            if not runs:
                continue
            if len(runs) > 1:
                ax.scatter(runs, [y] * len(runs), s=22, color=color, alpha=0.45, lw=0, zorder=4)
            means[(name, word)] = mean = float(np.mean(runs))
            ax.scatter([mean], [y], s=120, color=color, edgecolor=SURFACE, lw=2, zorder=6)
            texts.append(ax.text(mean, y + 0.07, f"{mean:+.2f}", ha="center", va="bottom",
                                 fontsize=10, color=INK2, zorder=7,
                                 bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85)))
            points.append((mean, y))

    # Join each word's positions from one period to the next
    for word, color in zip(WORDS, COLORS):
        present = [n for n in names if (n, word) in means]
        ax.plot([means[(n, word)] for n in present], [rows[n] for n in present],
                color=color, lw=2, alpha=0.5, solid_capstyle="round", zorder=5)

    ax.axvline(0, color=BASE, lw=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.32, len(names) - 1 + 0.55)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(axis="x", colors=MUTED, length=0, pad=6)
    ax.set_xlabel("←  leans toward the men words                 leans toward the women words  →",
                  fontsize=11, color=INK2, labelpad=10)

    quoted = [f"'{w}'" for w in WORDS]
    words_text = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    fig.text(0.15, 1 - 0.32 / height, f"{words_text} on each period's women–men ruler",
             fontsize=17, fontweight="bold", color=INK, va="bottom")
    runs_note = (f" Small dots are separate training runs ({EXTRA_RUNS + 1} per period)."
                 if EXTRA_RUNS else "")
    fig.text(0.15, 1 - 0.43 / height,
             "Each ruler is built inside its own period's model from that period's women and men words,\n"
             "so the periods can be compared without aligning the models." + runs_note,
             fontsize=11, color=INK2, va="top", linespacing=1.45)

    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=w) for w, c in zip(WORDS, COLORS)]
    if EXTRA_RUNS:
        handles.append(Line2D([], [], marker="o", ls="", ms=5, color=MUTED, alpha=0.6, label="one training run"))
    handles += [Line2D([], [], marker="D", ls="", ms=5, color=MUTED, label="ends of the ruler"),
                Patch(color=FILL, label=f"all words used {MIN_COUNT}+ times")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.145, 1 - 1.04 / height),
               frameon=False, fontsize=10, ncol=len(handles), columnspacing=1.6, handletextpad=0.4)

    # Nudge the labels apart so they don't overlap; a thin line joins a moved label to its dot
    adjust_text(texts, x=[x for x, _ in points], y=[y for _, y in points], ax=ax, min_arrow_len=20,
                arrowprops=dict(arrowstyle="-", color="#8a93a6", lw=0.6))

    PLOT_FOLDER.mkdir(exist_ok=True)
    path = PLOT_FOLDER / f"ruler_{'_'.join(WORDS)}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"\nSaved the plot to {path.relative_to(SCRIPT_DIR)}")
    if OPEN_PLOTS:
        webbrowser.open(path.resolve().as_uri())


def main():
    if not 1 <= len(WORDS) <= len(COLORS):
        print(f"Set WORDS to between 1 and {len(COLORS)} words.")
        return

    periods = {}
    for name in MODEL_NAMES:
        path = MODEL_FOLDER / f"{name}.model"
        if not path.exists():
            print(f"Can't find {path.relative_to(SCRIPT_DIR)}. Train it first with 01_train_model.py.")
            return
        periods[name] = measure(Word2Vec.load(str(path)), name)

    print("\nWhere each word falls on each period's ruler  (+ leans toward the women words, - toward the men words)")
    for name, p in periods.items():
        print(f"\n{name}  ({p['size']:,} words; the ruler runs from the men words at {p['men_end']:+.2f} "
              f"to the women words at {p['women_end']:+.2f})")
        for word in WORDS:
            runs = [run[word] for run in p["runs"] if run[word] is not None]
            if not runs:
                print(f"   {word:15} isn't in this period's vocabulary (too rare there)")
                continue
            spread = f"   runs: {min(runs):+.2f} to {max(runs):+.2f}" if len(runs) > 1 else ""
            print(f"   {word:15} {np.mean(runs):+.2f}{spread}   ({p['counts'][word]:,} uses)")

    plot_rulers(periods)


if __name__ == "__main__":
    main()
