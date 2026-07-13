# face_recognition — hpopt results

Best configurations found by the Optuna leave-one-out search. See `../../README.md`
for how the search works and what `fbeta` / `precision` / `recall` / `specificity`
mean.

# deepface

## Run 1
Find logs at: ./wandb/run-20260623_132044-raxq84l2/logs
[41:52<00:00, 12.56s/it]
Best trial #112: fbeta=0.7066, precision=0.7935, recall=0.4912, specificity=0.8598, n_known_queries=399, n_negative_queries=164, n_emitted_labels=247, 
params={'model_name': 'Facenet512', 'detector_backend': 'retinaface', 'distance_metric': 'cosine', 'threshold_cosine': 0.269106274490387}

# insightface

## Run 1
Find logs at: ./wandb/run-20260623_135827-q6cwmzva/logs
250/250 [28:26<00:00,  6.82s/it]
Best trial #140: fbeta=0.8423, precision=0.8432, recall=0.8389, specificity=0.8092, n_known_queries=391, n_negative_queries=152, n_emitted_labels=389, 
params={'model_pack': 'antelopev2', 'det_size': 480, 'pad_ratio': 0.4, 'distance_metric': 'cosine', 'threshold_cosine': 0.5437924395389178}

## Run 2
Notes: 
- we made some changes to the objective, to punish embedding failures
- added support for code carbon
- reworked the ground truth set slightly (only give id's for faces that are recognizable by the crop alone)
~ 2h 23 min
wandb: Find logs at: ./wandb/run-20260624_205709-86f31ghf/logs
Best trial #634: fbeta=0.8582, precision=0.8561, recall=0.8667, specificity=0.8408, n_known_queries=405, n_negative_queries=157, n_emitted_labels=410, 
params={'model_pack': 'antelopev2', 'det_size': 480, 'pad_ratio': 0.4, 'det_thresh': 0.1, 'distance_metric': 'euclidean_l2', 'threshold_euclidean_l2': 1.096825324690873}

### Run 2 — robustness note (1000 trials)

Reproduce with `uv run python scripts/analyze_hpopt_runs.py` (parses every run's
`config.yaml` + `wandb-summary.json` into one table).

The surprisingly low `det_thresh=0.1` is **not an overfit artifact** — it's a
near-irrelevant knob that the best-of-1000 selection happened to surface at 0.1:

- **The optimum is a plateau, not a peak.** 190 trials land within 0.005 of the
  best fbeta (0.8582), 387 within 0.01. The top three trials all score 0.8582
  with `det_thresh` = 0.1, 0.1 **and 0.3** — same score, different threshold.
- **`det_thresh` is flat (read the median, not the mean).** Holding the winning
  family fixed (`antelopev2 / det_size=480 / euclidean_l2`), median fbeta stays
  0.845–0.850 across the whole `det_thresh` range 0.1→0.6, and max barely moves
  (0.8559–0.8582). The low *mean* at 0.1 is just bad metric/threshold pairings,
  not the detector floor. Mechanically sensible: inputs are already tight face
  crops, so a low floor mostly avoids the detector rejecting its own crop;
  antelopev2 detects them at any threshold. **Pin `det_thresh` anywhere in
  0.1–0.4 with no measurable loss.**
- **What *is* robustly selected (trust these):** among the top 50 trials, 50/50
  `antelopev2`, 50/50 `euclidean_l2`, 49/50 `det_size=480`, 48/50 `pad_ratio=0.4`,
  and `threshold` clustered tight at 1.091 ± 0.005.
- **Caveat — selection bias, not `det_thresh`.** All 1000 trials are scored on
  the *same* ~562-query LOO set with no held-out split, so the headline 0.8582 is
  optimistically biased. But the gap to the 95th percentile is only ~0.002 and
  the winner sits inside a 190-trial cluster of near-identical configs, so the
  chosen config is robust — just don't read 0.8582 as a precise held-out estimate.