# Week 12: Network Graphs

An interactive kinship network of 517 enslaved people at George Washington's Mount Vernon, 1754–1799. It is a Python port of Sara Collini's R visualization, which was built with igraph and visNetwork.

## Files in this folder:
- `mv_social_network.py`: builds the network and writes the interactive page
- `data/nodes_040816.csv`: one row per person (`id`, `Census`, `Gender`, `location`)
- `data/edges_040816.csv`: one row per family relationship (`id`, `id.1`, `relationship`)
- `requirements.txt`: dependencies
- `mv_social_network.html`: the output (created when you run the script; open it in a browser)

## Setup Instructions:

### First Time Setup (Create New Virtual Environment):

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script:**
   ```bash
   python mv_social_network.py
   ```

5. **Open the result:**
   ```bash
   open mv_social_network.html
   ```

6. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Returning to Existing Virtual Environment:

```bash
source venv/bin/activate
python mv_social_network.py
deactivate
```

## R vs. Python

| Job | R | Python |
|---|---|---|
| Load and prepare the tables | `read.csv`, `dplyr` | `pandas` |
| Build the graph, compute layout and stats | `igraph` | `networkx` |
| Draw the interactive graph | `visNetwork` | `pyvis` |

Both `visNetwork` and `pyvis` are wrappers around the same JavaScript library, [vis.js](https://visjs.github.io/vis-network/docs/network/), so most of the styling options are the same. Some specific translations:

| R | Python |
|---|---|
| nested `ifelse()` to assign colors | a dictionary plus `.map()` |
| `visIgraphLayout(layout = "layout_with_fr", randomSeed = 12)` | `nx.spring_layout(G, seed=12)` |
| `visOptions(highlightNearest = ...)` | `Network(neighborhood_highlight=True)` |
| `visOptions(nodesIdSelection = TRUE)` | `Network(select_menu=True)` |
| `visOptions(selectedBy = "location")` | `Network(filter_menu=True)` (filters on any node field) |
| `visNodes()`, `visEdges()`, `visInteraction()` | `net.set_options(...)` with the same settings as JSON |
| `visLegend()` | no equivalent; the script writes a small HTML legend |

## What it does:
1. Loads the people (nodes) and relationships (edges).
2. Colors each person by farm (gray if they were not on the 1799 census) and shapes them by gender. Colors each line by relationship type, with dashed lines for "suspected" relationships.
3. Builds the graph in networkx and prints summary statistics: the number of family groups, the number of people with no recorded kin, and the most-connected people.
4. Lays out the graph with the Fruchterman-Reingold algorithm and draws it with pyvis.

In the page, click a person to highlight their family. Search for a name to zoom to that person, or use the filter menu to highlight a farm.

## Credits
Data from the Mount Vernon Slavery Database (Microsoft Access version), April 8, 2016. Original R visualization by Sara Collini; see her [blog post](http://www.saracollini.org/exploring-the-mount-vernon-enslaved-community-using-network-analysis/) on the project. Learn more about these individuals in the [Mount Vernon Slavery Database](http://www.mountvernon.org/george-washington/slavery/slavery-database/).
