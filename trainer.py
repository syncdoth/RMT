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
    remove_unused_columns: bool = False  # keep to False
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
    memory_gate_type: str = field(
        default='none',
        metadata={
            "help":
                "how to connect memory of different segments. Defaults to None (no connection).",
            "choices": ['none', 'residual', 'zero_conv', 'attention']
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
        inputs = self._prepare_inputs(inputs)
        model.eval()
        model.config.use_cache = True  # faster
        with torch.no_grad():
            logits = model(**inputs).logits
            labels = inputs['labels']
            # [B, T]
            loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
            eval_loss = loss.mean()
        if prediction_loss_only:
            return eval_loss
        # loss, logit, label
        # NOTE: quick hack to include session id in compute_metric: add it into labels.
        session_ids = inputs['session_ids']  # [B, 1]
        labels_session_ids = torch.cat([labels, session_ids], dim=1)
        # NOTE: for efficiency, compute ppl here!
        label_mask = labels != -100  # TODO -100 is pad idx by default;
        true_loss = loss.sum(-1) / label_mask.sum(-1)  # [B,]
        return (eval_loss, true_loss, labels_session_ids)


def compute_metrics(eval_preds):
    """Compute perplexity"""
    # NOTE: see prediction_step; predictions is actually "true loss" (loss concerning real seqlen)
    # This can be directly used for ppl computation
    loss = torch.tensor(eval_preds.predictions)  # [N,]
    labels = torch.LongTensor(eval_preds.label_ids)
    session_ids = labels[:, -1]  # [N,]  NOTE see prediction_step for explanation
    labels = labels[:, :-1]

    session_ppl = {}
    for i in torch.unique(session_ids):
        if i == -100:
            continue
        session_ppl[f'session-{i}-ppl'] = torch.exp(loss[session_ids == i].mean())

    total_ppl = torch.exp(loss.mean())
    session_ppl['all-ppl'] = total_ppl

    return session_ppl
