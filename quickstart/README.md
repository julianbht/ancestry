# Quickstart dummy dataset — Curie / Joliot-Curie

A tiny public stand-in for the private data, so a fresh clone can run the
pipeline and web app without any real family photos. Everyone here is a public
historical figure; genealogy facts are public. See `docs/going-public.md` for
the private/public split this belongs to.

`quickstart/data/` is a drop-in **data root**: point the project at it with the
`ANCESTRY_DATA_DIR` env var and everything below is read from here instead of
the real (gitignored) `data/` tree.

```
quickstart/data/
  originals/album/             # ignored — source photos copied aside
  raw/album/                     # you add the photos here
  gramps/database/family-tree-data.csv     # authored — the family tree
  curated/
    face_annotation/ground_truth.json      # empty skeleton — you fill
    frame_crop/ground_truth.json           # empty skeleton — you fill
    frame_crop/table_layouts.json          # generated placements + reusable metadata
    rotate/rotations.csv                   # header only — you fill
    photo_backs.csv                        # header only — you fill
```

## The family (`gramps/database/family-tree-data.csv`)

11 people across four generations, two intermarried lines — enough to exercise
grandparents, aunts/uncles, cousins, and in-laws in the kinship labels:

```
  Eugène Curie ─┬─ Sophie-Claire        Władysław ─┬─ Bronisława
                │                        Skłodowski │   Skłodowska
          Pierre Curie ──────┬────────── Marie Curie
                             │
              ┌──────────────┴───────────────┐
        Irène Joliot-Curie ─┬─ Frédéric   Ève Curie
                            │   Joliot-Curie
                 ┌──────────┴───────────┐
          Hélène Joliot-Curie    Pierre Joliot-Curie
```

The CSV is a Gramps "CSV export" (stacked `Place` / `Person` / `Marriage` /
`Family` sections). It's authored directly here — no `data.gramps` native file
is needed, since only the CSV is read by the code.

## Run it

```bash
# family tree + web app on the dummy tree
ANCESTRY_DATA_DIR=quickstart/data uv run ancestry-web

# or a pipeline step (once you've added photos to raw/album/)
ANCESTRY_DATA_DIR=quickstart/data uv run ancestry-rotate

# build the table scenes + frame-crop ground truth from the source album
uv run python scripts/quickstart/compose_album_on_table.py
```

`config/` is shared, public, and always read from the repo root — only the data
root switches.
