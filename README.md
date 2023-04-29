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
| BST 2.7B                 | 10.533    | 11.947    | 11.169    | 11.018    | 12.4      | 11.201 |
| RMT                      |           |           |           |           |           |        |
| BST 2.7B + RMT-base      |           |           |           |           |           |        |
| BST 2.7B + RMT-attention |           |           |           |           |           |        |
| BST 2.7B + RMT-residual  |           |           |           |           |           |        |

### Ablations

1. Session 5 test set perplexity with varying number of segments during train and test.

| Train \ Eval | 2 (235) | 4 (471) | 8 (943) | 16 (1,887) | all (max=3099) |
|--------------|---------|---------|---------|------------|----------------|
| 2            |         |         |         |            |                |
| 4            |         |         |         |            |                |
| 8            |         |         |         |            |                |
| 16           |         |         |         |            |                |

### Stats

Input sequence lengths

| split | max   | mean    | std     |
|-------|-------|---------|---------|
| train | 2331  | 741.150 | 513.017 |
| valid | 3132  | 973.731 | 654.354 |
| test  | 3099  | 981.203 | 656.987 |