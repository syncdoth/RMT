## Setup

```bash
pip install -r requirements.txt
```

Of course, managing the environment with conda is always encouraged. Notice that
the code utilizes `8-bit` quantization and lora, which requires at least Ampere
cuda-enabled GPU. (RTX 30xx and newer) The environment has been tested with
cuda-11.7 with cudnn-[7|8].x.

## Train

```bash
# big model; for detail, look at train.sh
sh train.sh $GPU_ids $memory_gate $num_seg left $train_bs $grad_check $num_seg 5
```

## Experiments

Take a look at scripts in `scripts/`.

* `baseline_eval.sh` - evaluates the baseline (BST 2.7B) model zero-shot.
* `eval-generate.sh` - generates response for the target test set session with `infer.py`
* `eval-seg*.sh` - evaluate the model with different segments during evaluation. `-small` is for 400M model.
* `experiment-seg*.sh` - train the model with different segments. `-small` for 400M model.
* `train-baseline.sh` - trains the baseline model (BST 2.7B / BST 400M) on MSC dataset.

## Results

* Test set perplexity.

config:
* memory length 10
* blenderbot-3B (lora)

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

config:
* memory length 10
* blenderbot-3B (lora)
* none, attention

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


### Remembering Previous Session

config:
* memory length 5
* blenderbot-400M-distill
* residual v2
* curriculum training (continue seg4 training from seg2 checkpoint, etc.)

After session 5, session 4's last query and response is asked again. Since the
query and response are both in the input history, the model should learn this task
well if it has enough effective memory size, which is around 650 on average. This
means that with 6 segments (737 tokens), the model always has the answer in
its input history.


| Effective Input Ctx | Model          | session 5 | remember task |
|---------------------|----------------|-----------|---------------|
|       128           | MSC 400M (128) |  14.004   | 16.077        |
|       123           | RMT 1          |  13.863   | 15.996        |
|       246           | RMT 2          |  13.683   | 15.945        |
|       492           | RMT 4          |  13.573   | 15.744        |
|       984           | RMT 8          |  13.567   | 15.721        |

* Automatic Metrics

|   session     | model    |   bleu1 |   bleu2 |   bleu3 |   bleu4 |   dist1 |   dist2 |   dist3 |   dist4 |
|--------------:|:---------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
|         5     | MSC 400M |   0.439 |   0.238 |   0.12  |   0.07  |   0.006 |   0.049 |   0.174 |   0.324 |
|         5     | RMT 1    |   0.439 |   0.24  |   0.119 |   0.07  |   0.006 |   0.048 |   0.172 |   0.319 |
|         5     | RMT 8    |   0.446 |   0.248 |   0.126 |   0.076 |   0.005 |   0.048 |   0.175 |   0.325 |
| Remember Task | MSC 400M |   0.468 |   0.242 |   0.113 |   0.063 |   0.006 |   0.049 |   0.175 |   0.322 |
| Remember Task | RMT 1    |   0.476 |   0.25  |   0.118 |   0.065 |   0.006 |   0.048 |   0.175 |   0.318 |
| Remember Task | RMT 8    |   0.482 |   0.259 |   0.124 |   0.069 |   0.006 |   0.05  |   0.178 |   0.324 |

### Stats

Input sequence lengths

| split | max   | mean    | std     |
|-------|-------|---------|---------|
| train | 2331  | 741.150 | 513.017 |
| valid | 3132  | 973.731 | 654.354 |
| test  | 3099  | 981.203 | 656.987 |
