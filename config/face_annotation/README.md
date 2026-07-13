# Annotating the face-recognition ground truth

`ground_truth.json` is scored by the leave-one-out Fβ objective in
`src/pipeline/experiments/face_recognition/`. The recognizer only ever sees the
**pixels of a face crop** — never your knowledge of who's in the photo. Annotate
accordingly.

## The one rule

> **Assign a `person_id` only if the face is identifiable from the crop itself.**
> Otherwise leave `person_id: null` (still draw the box).

"Identifiable from the crop" means a recognizer could plausibly match it from the
face alone. If you only know who it is from context (the scene, who else is
present, a side-on sliver you recognize), that's **not** identifiable — leave it
`null`. A bad call here doesn't just add noise to the objective; a low-quality
labeled crop also becomes a polluting reference in the production gallery.

## How each annotation affects the objective

A face only enters the eval if its box overlaps a `face_crop` detection
(overlap ≥ 0.30). Once matched:

| Annotation | Role | Effect on objective |
|---|---|---|
| `person_id` set, person has **≥ 2** labeled faces | **known query** | feeds **recall**: a hit on the right person is a true positive; a miss (incl. failed embedding) or wrong-person hit lowers recall |
| `person_id` set, person has **only 1** labeled face | negative query | can't validate its own identity, but still acts as a reference for others |
| `person_id: null` | negative query | only ever **lowers precision**, and only if wrongly labeled; a failed embedding or a correct "unknown" costs nothing |
| empty image (`faces: []`) | — | **inert**: contributes nothing in any direction |

Key asymmetry: a **labeled ≥2-photo face that fails to embed is an
unconditional recall penalty** (this is the pressure that pushes detector/embed
hyperparameters to reduce failures). A **null face is a fair test** — it charges
you only for a real false positive, never automatically.

## Edge cases

- **Face present but unidentifiable from the crop** (tiny, side-on, blurry): draw
  the box, `person_id: null`. Don't label it for anyone — labeling it for a
  well-represented person guarantees a recall miss no hyperparameter can fix.
- **Identifiable only because you know the context**: still `null`. Your context
  doesn't transfer to the recognizer.
- **No annotatable face at all**: leave `faces: []`. This is the maximally safe
  choice — it can never affect the score, even if `face_crop` falsely detected a
  face there.
- **Don't know the person**: `person_id: null`. Never invent an id and never use a
  literal string like `"unknown"` — only JSON `null` marks a face unidentified.
  Any non-null string is treated as a real person and corrupts both the eval and
  the gallery.

## `age`

`age` does not affect the face-recognition objective. Fill it in where known for
the downstream age-estimation step; leave `null` otherwise.
