# frame_crop — hpopt results

Best configurations found by the Optuna search. See `../../README.md` for how the
search works and the parameter ranges. Per-run CSV/SVG exports live alongside this file.

## Canny (`--method canny`)

**Run 1**
Note: this run had a bug in `threshold_1` and `threshold_2`.
```
[19:47<00:00, 7.92s/it]
Best trial #141: accuracy=0.5882, mean_iou=0.5541,
params={'threshold1': 109.54213261959288, 'threshold2': 38.47619525286581,
        'blur_kernel_size': 11, 'morph_kernel_size': 7, 'morph_iterations': 14}
```

**Run 3**
Note: we renamed `accuracy` to `detection_rate`. The formula is unchanged — the name
just better reflects the metric.
```
[21:40<00:00, 8.67s/it]
Best trial #110: detection_rate=0.5980, mean_iou=0.5553,
params={'threshold_low': 65.2626896385325, 'threshold_delta': 33.952014812487434,
        'blur_kernel_size': 11, 'morph_kernel_size': 7, 'morph_iterations': 15}
```
Export: `canny-frame-crop-experiment-3.csv`, `canny-frame-crop-experiment-3.svg`

## SAM 3.1 (`--method sam`)
```
[1:24:33<00:00, 101.47s/it]
Best trial #15: detection_rate=0.9902, mean_iou=0.9487,
params={'prompt': 'rectangular photograph', 'score_threshold': 0.6781474901164721}
```
Export: `sam-3-frame-crop-experiment-1.csv`, `sam-3-frame-crop-experiment-1.svg`,
`sam_prompt_bar.png`
