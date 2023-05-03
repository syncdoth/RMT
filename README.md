## Train

```bash
GPU_ID=1  # gpu id to use
MEMORY_GATE=none  # choose from ['none', 'attention', 'residual']
N_SEGMENTS=4   # number of segments to train RMT model with. N_SEGMENT=4 means the effective
# input context size for the encoder is
#        (max_input_length - memory_length) * N_SEGMENT - 1,
# where memory_length is number of memory tokens used and -1 at the end is for EOS.

sh train.sh $GPU_ID $MEMORY_GATE $N_SEGMENTS
```

## Results

* Test set perplexity.

| model                    | session 1 | session 2 | session 3 | session 4 | session 5 |  all   |
|--------------------------|-----------|-----------|-----------|-----------|-----------|--------|
| Baseline                 |           |           |           |           |           |        |
| BST 2.7B (our)           | 11.513    | 10.775    | 11.182    | 11.238    | 11.3      | 11.201 |
| BST 2.7B (paper)         | 8.97      | 9.98      | 10.26     | 10.40     | 10.5      | -      |
| MSC 2.7B (paper)         | 8.87      | 8.89      | 9.10      | 9.21      | 9.27      | -      |
| MSC 2.7B (paper)         | 8.25      | 8.76      | 8.93      | 9.07      | 9.16      | -      |
| RMT                      |           |           |           |           |           |        |
| MSC 2.7B + RMT-base      | 7.42      | 8.476     | 8.88     | 8.984      | 9.207     | 8.563  |
| MSC 2.7B + RMT-attention | 7.632     | 8.565     | 8.866     | 8.767     | 8.937     | 8.535  |
| MSC 2.7B + RMT-residual  | 8.38      | 9.397     | 9.615     | 9.642     | 9.926     | 9.371  |

* Validation perplexity

| model                    | session 1 | session 2 | session 3 | session 4 | session 5 |  all   |
|--------------------------|-----------|-----------|-----------|-----------|-----------|--------|
| Baseline                 |           |           |           |           |           |        |
| BST 2.7B                 | 11.410    | 11.265    | 11.397    | 11.272    | 11.257    | 11.324 |


### Ablations

1. Session 5 test set perplexity with varying number of segments during train and test.

| Train \ Eval | 2 (235) | 4 (471) | 8 (943) | 16 (1,887) | all (max=3099) |
|--------------|---------|---------|---------|------------|----------------|
| RMT 2        |  8.85   | 8.85    |  8.85   |  9.718     |    10.699      |
| RMT 4        |  8.884  | 8.884   |  8.884  |  9.26      |     9.72       |
| RMT 8        |  8.951  | 8.951   |  8.951  |  9.046     |     9.183      |
| RMT 16       |  9.242  | 9.232   |  9.15   |  8.99      |     9.207      |
| RMT 2 att    |  8.805  | 8.859   |  34.34  |  2245.464  |    3233.652    |
| RMT 4 att    | 464.365 | 8.785   |  9.41   |  2519.306  |   2328.181     |
| RMT 8 att    | 659.919 | 638.065 |  8.8    |  1588.278  |   1350.88      |
| RMT 16 att   | 956.655 | 874.815 |  10.294 |  8.898     |     8.937      |

### Stats

Input sequence lengths

| split | max   | mean    | std     |
|-------|-------|---------|---------|
| train | 2331  | 741.150 | 513.017 |
| valid | 3132  | 973.731 | 654.354 |
| test  | 3099  | 981.203 | 656.987 |
