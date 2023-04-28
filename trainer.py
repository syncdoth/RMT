from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import TrainingArguments
from transformers.trainer import Trainer


@dataclass
class RMTTrainingArgs(TrainingArguments):
    # training args
    learning_rate: float = 5e-4
    warmup_steps: int = 1000
    weight_decay: float = field(default=0, metadata={"help": "typically, set this to 0.01"})
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    evaluation_strategy: str = 'steps'
    eval_steps: int = 1000
    max_steps: int = 100000
    dataloader_drop_last: bool = False
    report_to: str = 'wandb'
    output_dir: str = 'outputs'
    logging_steps: int = 100
    save_strategy: str = 'steps'
    save_steps: int = 1000
    dataloader_num_workers: int = 0  # TODO
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    ################################# RMT ARGS #################################
    memory_length: int = 10
    num_segments: int = 1
    memory_position: str = field(default='left',
                                 metadata={
                                     "help": "where the memory tokens to be placed.",
                                     "choices": ['left', 'right', 'split']
                                 })
    write_memory_position: str = field(default='right',
                                       metadata={
                                           "help": "where the memory tokens to be placed.",
                                           "choices": ['left', 'right', 'split']
                                       })


class RMTTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs).loss

        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
            labels = inputs['labels']
            # [B, T]
            loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
            loss = loss.mean()
        if prediction_loss_only:
            return loss
        # loss, logit, label
        return (loss, logits, labels)


def compute_metrics(eval_preds):
    """Compute token accuracy of greedy decoding"""
    pred = np.argmax(eval_preds.predictions, axis=-1)
    num_correct = (pred == eval_preds.label_ids).sum()
    num_predict = pred.size

    return {'accuracy': num_correct / num_predict}
