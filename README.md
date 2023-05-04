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


| Train \ Eval   | session 5 | remember task |
|----------------|-----------|---------------|
| MSC 400M (128) |  14.004   | 16.077        |
| RMT 1          |  13.863   | 15.996        |
| RMT 2          |  13.683   | 15.945        |
| RMT 4          |  13.573   | 15.744        |
| RMT 8          |  13.567   | 15.721        |

* Other Automatic

```
session 5
# baseline
{'dist1': 0.005835076510145353, 'dist2': 0.04887008788949128, 'dist3': 0.17449059982608298, 'dist4': 0.3235796421227604, 'bleu1': 0.4387617434246656, 'bleu2': 0.23849
133880273946, 'bleu3': 0.11995139355161161, 'bleu4': 0.06964186204772198}

# seg1
{'dist1': 0.005791083909756355, 'dist2': 0.04772425977935908, 'dist3': 0.17158176924532878, 'dist4': 0.3190239490295709, 'bleu1': 0.4390971784908739, 'bleu2': 0.24000
911615776999, 'bleu3': 0.1191760917854938, 'bleu4': 0.06952772616555128}

# seg8
{'dist1': 0.005491604176268345, 'dist2': 0.047924004645187424, 'dist3': 0.17531556783329866, 'dist4': 0.32522630567649013, 'bleu1': 0.4458228226102809, 'bleu2': 0.248
11986721223986, 'bleu3': 0.12612740155098615, 'bleu4': 0.07550621254148358}

session 200
# baseline
{'dist1': 0.006204535722930642, 'dist2': 0.049199826936418986, 'dist3': 0.17544818520392644, 'dist4': 0.3215075151176974, 'bleu1': 0.4684643772862806, 'bleu2': 0.24243153731546674, 'bleu3': 0.11312916743544736, 'bleu4': 0.06276318731103822}

# seg1
{'dist1': 0.005535055344546391, 'dist2': 0.0479482114900723, 'dist3': 0.17506101600281682, 'dist4': 0.31826340552135596, 'bleu1': 0.4763608295680607, 'bleu2': 0.2504073474724789, 'bleu3': 0.1180694698607399, 'bleu4': 0.0654248556984865}

# seg8
{'dist1': 0.006064281376527005, 'dist2': 0.050234837604416244, 'dist3': 0.17835774687604947, 'dist4': 0.3237440063333918, 'bleu1': 0.48153538550882097, 'bleu2': 0.2594836951905992, 'bleu3': 0.12370669068085921, 'bleu4': 0.06925412796157368}
```

### Stats

Input sequence lengths

| split | max   | mean    | std     |
|-------|-------|---------|---------|
| train | 2331  | 741.150 | 513.017 |
| valid | 3132  | 973.731 | 654.354 |
| test  | 3099  | 981.203 | 656.987 |
