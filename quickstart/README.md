# Quickstart demo — the Curie / Joliot-Curie family

A small, ready-to-run dataset that lets a fresh clone exercise the whole
pipeline and the web app without any real photos. Everyone here is a public
historical figure and the genealogy is public record, so it ships in the repo.

It is **fully populated**: 25 source photos, hand-labelled ground truth for
faces and print frames, and the pipeline's precomputed outputs.

`quickstart/data/` is a drop-in **data root**: set `ANCESTRY_DATA_DIR` to it and
every step reads from here instead of the real (gitignored) `data/` tree.
`config/` is shared and always read from the repo root — only the data root
switches.

## What's inside

```
quickstart/data/
  raw/album/                            # 25 source photos (public-domain Curie portraits)
  gramps/database/family-tree-data.csv  # the family tree (Gramps CSV export)
  curated/
    face_annotation/ground_truth.json   # face boxes + person IDs + ages
    frame_crop/ground_truth.json        # print-quad labels
    frame_crop/table_layouts.json       # scene placements + reusable metadata
    rotate/rotations.csv                # rotation overrides (none needed for this set)
    photo_backs.csv                     # front → back-note mapping (none for this set)
  steps/                                # precomputed outputs: frame_crop, face_crop, face_recognition
  state/                                # per-step progress JSON
  label_studio/                         # generated annotation tasks + a project export
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
`Family` sections), authored directly here — the code only reads the CSV, so no
native `data.gramps` file is needed.

## Run it

`.env.example` already sets `ANCESTRY_DATA_DIR=quickstart/data`, so pass it with
`--env-file` (works the same in bash and PowerShell):

```bash
# family tree + web app on the dummy tree
uv run --env-file .env.example --group web ancestry-web

# re-run a pipeline step (already-processed files are skipped)
uv run --env-file .env.example ancestry-rotate
```

The committed `steps/` outputs mean the web app has something to show
immediately; re-running a step just reproduces them.
