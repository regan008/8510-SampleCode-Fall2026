#!/usr/bin/env python3
"""
Social Network of the Enslaved Community at Mount Vernon, 1754-1799

A Python port of Sara Collini's R visualization (mv_social_network.qmd),
which used igraph + visNetwork. The Python equivalents:

    R                    Python      Job
    -------------------  ----------  ------------------------------------
    read.csv + dplyr     pandas      load and prepare the tables
    igraph               networkx    build the graph, compute the layout
    visNetwork           pyvis       draw the interactive graph

visNetwork and pyvis are both wrappers around the same JavaScript library
(vis.js), so the styling options carry over almost word for word.

Run:     python mv_social_network.py
Output:  mv_social_network.html  (open it in a web browser)

Author: Created for History 8510 at Clemson University
"""

from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

HERE = Path(__file__).parent
DATA = HERE / "data"
OUTPUT = HERE / "mv_social_network.html"


# ---------------------------------------------------------------------------
# 1. Read in the data
# ---------------------------------------------------------------------------
# nodes: one row per person          columns: id, Census, Gender, location
# edges: one row per relationship    columns: id, id.1, relationship
nodes = pd.read_csv(DATA / "nodes_040816.csv")
edges = pd.read_csv(DATA / "edges_040816.csv")

# R: rename(from = id, to = id.1)
edges = edges.rename(columns={"id": "source", "id.1": "target"})

# Sanity check: every name in the edge list should also be in the node list.
# (networkx would quietly create a new, unstyled node for any typo.)
missing = (set(edges["source"]) | set(edges["target"])) - set(nodes["id"])
if missing:
    print("WARNING - names in edges but not in nodes:", sorted(missing))


# ---------------------------------------------------------------------------
# 2. Style the edges: color by relationship type
# ---------------------------------------------------------------------------
# In R this was a stack of nested ifelse() calls. In pandas we write the
# lookup as a dictionary and .map() it onto the column.
EDGE_COLORS = {
    "Spouse": "red",      "Suspected Spouse": "red",
    "Child": "darkcyan",  "Suspected Child": "darkcyan",
    "Sibling": "orange",  "Suspected Sibling": "orange",
}
edges["color"] = edges["relationship"].map(EDGE_COLORS).fillna("gray")

# New in this version: draw "Suspected" relationships as dashed lines
edges["suspected"] = edges["relationship"].str.startswith("Suspected")


# ---------------------------------------------------------------------------
# 3. Style the nodes: color by farm, shape by gender
# ---------------------------------------------------------------------------
FARM_COLORS = {
    "Mansion House": "orange",
    "Muddy Hole": "lightcoral",
    "Union Farm": "yellowgreen",
    "River Farm": "royalblue",
    "Dogue Run": "mediumturquoise",
    "Mill Complex": "mediumturquoise",
    "Other Location": "gold",
}
SHAPES = {"Female": "dot", "Male": "square"}  # anything else -> triangle

# Only people on the 1799 census get a farm color; everyone else is gray.
# .where(condition, other) keeps the value where condition is True.
on_1799_census = nodes["Census"] == 1799
nodes["color"] = nodes["location"].map(FARM_COLORS).where(on_1799_census, "lightgray")
nodes["shape"] = nodes["Gender"].map(SHAPES).fillna("triangle")


# ---------------------------------------------------------------------------
# 4. Build the graph with networkx (R: igraph)
# ---------------------------------------------------------------------------
G = nx.Graph()

for person in nodes.itertuples():
    G.add_node(
        person.id,
        census=person.Census,
        gender=person.Gender,
        location=person.location,
        color=person.color,
        shape=person.shape,
    )

for rel in edges.itertuples():
    G.add_edge(
        rel.source,
        rel.target,
        relationship=rel.relationship,
        color=rel.color,
        suspected=bool(rel.suspected),
    )

# What networkx can tell us that the picture can't
families = [group for group in nx.connected_components(G) if len(group) > 1]
print(f"{G.number_of_nodes()} people, {G.number_of_edges()} relationships")
print(f"{len(families)} connected family groups; the largest has "
      f"{max(len(group) for group in families)} people")
print(f"{nx.number_of_isolates(G)} people have no recorded family ties")
print("\nMost connected people (number of recorded ties):")
for name, ties in sorted(G.degree, key=lambda pair: pair[1], reverse=True)[:10]:
    print(f"  {name:<32} {ties}")


# ---------------------------------------------------------------------------
# 5. Compute the layout (R: visIgraphLayout(layout = "layout_with_fr"))
# ---------------------------------------------------------------------------
# networkx calls the Fruchterman-Reingold algorithm "spring_layout".
# Fixing the seed (R: randomSeed = 12) gives the same picture on every run.
# k is the ideal distance between nodes: raise it to spread families out.
pos = nx.spring_layout(G, k=0.1, seed=12)
SCALE = 1500  # networkx returns coordinates between -1 and 1; vis.js wants pixels


# ---------------------------------------------------------------------------
# 6. Draw it with pyvis (R: visNetwork)
# ---------------------------------------------------------------------------
net = Network(
    height="750px",
    width="100%",
    neighborhood_highlight=True,  # R: visOptions(highlightNearest = ...)
    select_menu=True,             # R: visOptions(nodesIdSelection = TRUE)
    filter_menu=True,             # R: visOptions(selectedBy = "location")
    cdn_resources="remote",       # produce a single, self-contained .html file
)

for name, person in G.nodes(data=True):
    x, y = pos[name]
    net.add_node(
        name,
        label=name,
        title=f"{name}\n{person['location']}\nCensus: {person['census']}",  # hover text
        color=person["color"],
        shape=person["shape"],
        x=float(x * SCALE),
        y=float(y * SCALE),
        # Extra fields become choices in the filter menu
        location=person["location"],
        gender=person["gender"],
    )

for a, b, rel in G.edges(data=True):
    net.add_edge(
        a,
        b,
        title=rel["relationship"],                         # hover text
        color={"color": rel["color"], "opacity": 0.2},     # R: visEdges(color = list(..., opacity = 0.2))
        dashes=rel["suspected"],
    )

# Global vis.js options: the same settings as R's visNodes(), visEdges(),
# and visInteraction(), written as JSON instead of function arguments.
# Physics is off because networkx already decided where every node goes.
net.set_options("""
{
  "nodes": {"size": 8, "font": {"size": 10}},
  "edges": {"smooth": false, "hoverWidth": 1, "selectionWidth": 6},
  "interaction": {"hover": true, "hoverConnectedEdges": true,
                  "multiselect": true, "dragNodes": true},
  "physics": {"enabled": false}
}
""")


# ---------------------------------------------------------------------------
# 7. Add a title, legend, and credits, then save (R: visLegend)
# ---------------------------------------------------------------------------
# pyvis has no legend function, so we write a little HTML and paste it
# into the page that pyvis generates.
def swatch(color, shape="circle"):
    radius = "50%" if shape == "circle" else "0"
    return (f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:{radius};background:{color};margin-right:6px"></span>')

def line(color, dashed=False):
    style = "dashed" if dashed else "solid"
    return (f'<span style="display:inline-block;width:18px;border-top:3px {style} {color};'
            f'vertical-align:middle;margin-right:6px"></span>')

farm_rows = [("Mansion House", "orange"), ("Muddy Hole", "lightcoral"),
             ("Union Farm", "yellowgreen"), ("River Farm", "royalblue"),
             ("Dogue Run / Mill", "mediumturquoise"), ("Other Location", "gold"),
             ("Not on 1799 census", "lightgray")]

legend = "".join(f"{swatch(color)}{label}<br>" for label, color in farm_rows)
legend += "<br>"
legend += f"{swatch('gray')}Female<br>{swatch('gray', 'square')}Male<br>&#9650; Unknown<br><br>"
legend += f"{line('red')}Spouse<br>{line('darkcyan')}Child<br>{line('orange')}Sibling<br>"
legend += f"{line('gray', dashed=True)}Suspected"

HEADER = """
<div style="font-family: sans-serif; padding: 8px 16px;">
  <h2 style="margin: 0 0 4px;">Social Network of the Enslaved Community at Mount Vernon, 1754&ndash;1799</h2>
  <p style="margin: 0; font-size: 13px; max-width: 900px;">
    Each node is one person; each line is a known family relationship. Click a person to
    highlight their kin, or search for a name with the menu below. The capital letter after a
    first name only distinguishes people with the same name; it is not a surname.
  </p>
</div>
"""

LEGEND = f"""
<div style="position: fixed; bottom: 16px; left: 16px; background: white;
            border: 1px solid #ccc; border-radius: 6px; padding: 10px 12px;
            font: 12px sans-serif; line-height: 1.6; z-index: 10;">
  {legend}
</div>
"""

CREDITS = """
<p style="font: 11px sans-serif; text-align: center; color: #555; margin: 8px 16px;">
  Data from the Mount Vernon Slavery Database (Microsoft Access version), April 8, 2016.
  Original R visualization by Sara Collini. Python port for History 8510, Clemson University.
</p>
"""

# pyvis highlights a person picked from the search menu but doesn't move the
# view to them. This wraps pyvis's own selectNode() so it also zooms in.
ZOOM_TO_SELECTION = """
<script>
  const pyvisSelectNode = selectNode;
  selectNode = function (ids) {
    pyvisSelectNode(ids);
    network.focus(ids[0], {scale: 1.2, animation: true});
  };
</script>
"""

html = net.generate_html()
html = html.replace("<body>", "<body>" + HEADER, 1)
html = html.replace("</body>", LEGEND + CREDITS + ZOOM_TO_SELECTION + "</body>", 1)
OUTPUT.write_text(html, encoding="utf-8")

print(f"\nSaved {OUTPUT.name} - open it in a web browser.")
