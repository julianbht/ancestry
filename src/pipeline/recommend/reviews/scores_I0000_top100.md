# Recommender — manual evaluation

Whole-collection retrieval, reviewed depth 100. Source files:
- `recommend_I0000_max_top100.csv`
- `recommend_I0000_centroid_top100.csv`

## Precision@k

| Method | @10 | @25 | @50 | @100 |
| --- | ---: | ---: | ---: | ---: |
| max | 1.00 | 0.92 | 0.92 | 0.86 |
| centroid | 1.00 | 1.00 | 1.00 | 1.00 |

## nDCG@k

| Method | @10 | @25 | @50 | @100 |
| --- | ---: | ---: | ---: | ---: |
| max | 1.00 | 1.00 | 0.99 | 0.98 |
| centroid | 1.00 | 1.00 | 1.00 | 1.00 |
