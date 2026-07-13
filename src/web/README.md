# Family Photos — Web Frontend

A minimal local web app to browse the recognised relatives and their photos.

## Run

```bash
uv run ancestry-web
```

Then open <http://127.0.0.1:8000>.

## Pages

- **Homepage** (`/`) — one card per relative, grouped by family surname (families
  with the most photos first; within a family, the most-photographed person
  first). The portrait is the human `is_portrait` crop from
  `ground_truth.json`, falling back to a curated `data/gramps/portraits/` image, then
  to an initials placeholder.
- **Stammbaum** (`/tree`) — an interactive family tree as an alternative
  homepage. People are laid out by generation (hierarchical layout via the
  vendored [vis-network](https://visjs.github.io/vis-network/) library); drag to
  pan, scroll to zoom, click anyone with a photo to open their page. Each couple
  is joined through an invisible **union (marriage) node** half a generation
  below them, so a sibling group hangs from one shared point instead of crossing
  lines from both parents; childless couples keep a dashed spouse line.
  Generations are kept on the same vertical row (a child is always exactly one
  row below its parents). A switcher at the top of both home views
  (**Übersicht** / **Stammbaum**) toggles between them.
- **Person page** (`/person/{id}`) — every photo the person appears in, sorted
  by estimated age, with the age shown.
- **"Wer bist du?"** — pick yourself in the header and every person is labelled
  with their German kinship term relative to you (Großmutter, Onkel, Cousine,
  Urgroßvater, angeheiratet, …). Default viewer: `I0004` (Ève Curie in the quickstart tree).

### Turning the Stammbaum off

The whole family-tree feature sits behind one flag, `FAMILY_TREE_ENABLED` in
`web/config.py`. Set it to `False` and the `/tree` route 404s, the switcher tabs
disappear, and only the roster homepage remains. All tree code is confined to
`config.py`, `family_tree_graph.py`, `templates/tree.html`, and
`static/vendor/vis-network.min.js`.

The layout spacing is also config, in `FAMILY_TREE_SPACING` (`web/config.py`):
`level_separation` (vertical gap between rows — generations are ~2× this since a
union junction sits between them), `node_spacing` (horizontal gap within a row),
and `tree_spacing` (gap between branches). Changes take effect on page reload.

## Human vs. computed labels

Everything carries a flair so a viewer can tell a human annotation from a model
guess:

- <kbd>bestätigt</kbd> — from `data/curated/face_annotation/ground_truth.json` (human).
- <kbd>geschätzt</kbd> — from the face-recognition model output / kinship
  derivation (computed).

## Architecture (separation of concerns)

| Module            | Responsibility                                                        |
| ----------------- | -------------------------------------------------------------------- |
| `models.py`       | Domain dataclasses (`Person`, `Appearance`, `LabelSource`). No I/O.   |
| `family_tree.py`  | Parse the Gramps CSV into people + kinship edges (data access).       |
| `repository.py`   | Join ground truth + recognition sidecars + tree into the index.       |
| `kinship.py`      | Pure family-graph logic → German relationship terms.                  |
| `portraits.py`    | Crop/cache portraits, curated fallback (the only image I/O).          |
| `family_tree_graph.py` | Pure tree → node/edge graph (generation levels) for the Stammbaum. |
| `presenter.py`    | Build view models (grouping, ordering, label wiring).                 |
| `config.py`       | Feature flags (e.g. `FAMILY_TREE_ENABLED`).                           |
| `app.py`          | FastAPI routes — thin HTTP/templating layer only.                     |

Cropped portraits are cached under `data/web_cache/portraits/` (gitignored).
The index is built once at startup since the photo set is static within a run.
