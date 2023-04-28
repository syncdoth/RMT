from dataclasses import dataclass, field
import os

import wandb
from transformers import DataCollatorForSeq2Seq
from transformers.hf_argparser import HfArgumentParser
import torch

from data import MscDataset
from trainer import RMTTrainer, RMTTrainingArgs, compute_metrics
from model import load_transformer_LM_tokenizer, prepare_rmt_model


@dataclass
class ExperimentArgs:
    # facebook/blenderbot-3B
    model_name: str = field(default='facebook/blenderbot-400M-distill')
    tokenizer_name: str = field(default=None, metadata={"help": "Will default to --model_name."})
    # wandb logging
    wandb_project_name: str = "RMT"
    wandb_run_name: str = 'RMT-default-params'
    # peft
    use_lora: bool = field(
        default=False,
        metadata={"help": "use lora with huggingface peft. You must install loralib and peft."})
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    train_8bit: bool = False
    # data
    train_data_path: str = 'msc/session_4/train.txt'
    validation_data_path: str = 'msc/session_5/valid.txt'
    test_data_path: str = 'msc/session_5/test.txt'

    train_max_session: int = 1
    valid_max_session: int = 1
    test_max_session: int = 1

    test_only: bool = False


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # allows tf32, only on Ampere GPUs
    parser = HfArgumentParser([ExperimentArgs, RMTTrainingArgs])
    args, rmt_train_args = parser.parse_args_into_dataclasses()

    if rmt_train_args.deepspeed and args.train_8bit:
        raise ValueError("--train_8bit is not compatible with deepspeed.")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        device_map = {"": rmt_train_args.local_rank}
        if args.train_8bit:
            rmt_train_args.ddp_find_unused_parameters = False  # integral for train_8bit
    else:
        if args.train_8bit:
            device_map = 'auto'
        else:
            device_map = None

    if rmt_train_args.local_rank == 0:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config={
                "experiment": args.__dict__,
                "train": rmt_train_args.__dict__,
            },
        )

    model, tokenizer = load_transformer_LM_tokenizer(args.model_name,
                                                     tokenizer_name_or_path=args.tokenizer_name,
                                                     load_in_8bit=args.train_8bit,
                                                     device_map=device_map)
    prepare_rmt_model(model, tokenizer, rmt_train_args.memory_length,
                      rmt_train_args.memory_position, rmt_train_args.write_memory_position)

    if args.use_lora:
        from peft import (get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType,
                          prepare_model_for_int8_training)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        if args.train_8bit:
            model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        rmt_train_args.gradient_checkpointing = False  # incompatible with lora

    max_input_seq_len = (model.config.max_position_embeddings -
                         rmt_train_args.memory_length) * rmt_train_args.num_segments

    if not args.test_only:
        train_dataset = MscDataset(
            tokenizer,
            args.train_data_path,
            max_length=max_input_seq_len,
            memory_length=rmt_train_args.memory_length,
            memory_position=rmt_train_args.memory_position,
            max_session=args.train_max_session,
            mode='train',
        )
    else:
        train_dataset = None

    if rmt_train_args.evaluation_strategy != 'no':
        valid_dataset = MscDataset(
            tokenizer,
            args.validation_data_path,
            max_length=max_input_seq_len,
            memory_length=rmt_train_args.memory_length,
            memory_position=rmt_train_args.memory_position,
            max_session=args.valid_max_session,
            mode='eval',
        )
    else:
        valid_dataset = None
        rmt_train_args.eval_steps = None

    test_dataset = MscDataset(
        tokenizer,
        args.test_data_path,
        max_length=max_input_seq_len,
        memory_length=rmt_train_args.memory_length,
        memory_position=rmt_train_args.memory_position,
        max_session=args.test_max_session,
        mode='eval',
    )

    ################################################################

    trainer = RMTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=rmt_train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer,
                                             model,
                                             max_length=model.config.max_position_embeddings),
        compute_metrics=compute_metrics,
    )
    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
                model, type(model))

    if not args.test_only:
        trainer.train()
        model.save_pretrained(f"{rmt_train_args.output_dir}/{args.wandb_run_name}")
    trainer.evaluate(test_dataset, metric_key_prefix="test")
    wandb.finish()


if __name__ == "__main__":
    main()
