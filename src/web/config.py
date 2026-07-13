"""Web frontend feature flags.

Plain module-level constants — the web app is not a pipeline step, so it does
not use the Pydantic/YAML config machinery. Flip a flag here to change which
features the server exposes.
"""

from __future__ import annotations

# Show the Stammbaum (family-tree) homepage view and the view switcher.
# Set to False to hide the tree entirely: the /tree route 404s and the switcher
# tabs disappear, leaving only the original roster homepage.
FAMILY_TREE_ENABLED = False

# vis-network hierarchical-layout spacing for the Stammbaum, in pixels. Tune
# these to spread the tree out or pack it tighter; they are passed straight to
# the layout and take effect on the next page load.
FAMILY_TREE_SPACING = {
    # Vertical gap between layout rows. Person rows are two levels apart (a
    # union/marriage junction sits on the row between them), so the visible gap
    # between generations is roughly twice this value.
    "level_separation": 80,
    # Horizontal gap between neighbouring nodes within a row.
    "node_spacing": 95,
    # Horizontal gap between separate branches / sub-trees.
    "tree_spacing": 150,
}
