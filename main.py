from dataclasses import dataclass, field
import os

import wandb
from transformers import DataCollatorForSeq2Seq
from transformers.hf_argparser import HfArgumentParser
import torch

from data import MscDataset
from trainer import RMTTrainer, RMTTrainingArgs, compute_metrics
from model import load_transformer_LM_tokenizer


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
    load_baseline: bool = False

    load_checkpoint: str = field(default=None)
    load_peft_checkpoint: str = field(default=None)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # allows tf32, only on Ampere GPUs
    parser = HfArgumentParser([ExperimentArgs, RMTTrainingArgs])
    args, rmt_train_args = parser.parse_args_into_dataclasses()

    if rmt_train_args.deepspeed and args.train_8bit:
        raise ValueError("--train_8bit is not compatible with deepspeed.")
    if args.train_8bit:
        device_map = 'auto'
        rmt_train_args.ddp_find_unused_parameters = False  # integral for train_8bit
    else:
        device_map = None

    # 0 means main process in DDP training, -1 means simple single-gpu python launch
    if rmt_train_args.local_rank in (0, -1):
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config={
                "experiment": args.__dict__,
                "train": rmt_train_args.__dict__,
            },
        )

    model, tokenizer = load_transformer_LM_tokenizer(
        args.model_name,
        tokenizer_name_or_path=args.tokenizer_name,
        memory_length=rmt_train_args.memory_length,
        memory_position=rmt_train_args.memory_position,
        write_memory_position=rmt_train_args.write_memory_position,
        memory_gate_type=rmt_train_args.memory_gate_type,
        load_in_8bit=args.train_8bit,
        device_map=device_map,
    )

    if args.use_lora:
        from peft import (get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType,
                          prepare_model_for_int8_training, PeftModel)
        modules_to_save = ['embed_tokens']  # save embedding
        if rmt_train_args.memory_gate_type == 'attention':
            modules_to_save.append('memory_attention')
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            modules_to_save=modules_to_save,
        )
        if args.train_8bit:
            model = prepare_model_for_int8_training(model)
        if args.load_peft_checkpoint:
            model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)
        else:
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        rmt_train_args.gradient_checkpointing = False  # incompatible with lora

    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint), strict=False)

    # prepare max input seq length with number of segments for training.
    max_input_seq_len = (model.config.max_position_embeddings -
                         rmt_train_args.memory_length) * rmt_train_args.num_segments
    print(f"During training, {rmt_train_args.num_segments} segments will be used.",
          f"This leads to {max_input_seq_len - 1} tokens in encoder input.")

    # for eval, defaults to use the entire sequence.
    if rmt_train_args.eval_num_segments > 0:
        eval_max_input_seq_len = (model.config.max_position_embeddings -
                                  rmt_train_args.memory_length) * rmt_train_args.eval_num_segments
        print(f"During eval, {rmt_train_args.eval_num_segments} segments will be used.",
              f"This leads to {eval_max_input_seq_len - 1} tokens in encoder input.")
    else:
        print("During eval, entire sequence (history + query) will be used, no matter how long.")
        eval_max_input_seq_len = -1

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
        rmt_train_args.evaluation_strategy = 'no'  # skip loading validation set
        train_dataset = None

    if rmt_train_args.evaluation_strategy != 'no':
        valid_dataset = MscDataset(
            tokenizer,
            args.validation_data_path,
            max_length=eval_max_input_seq_len,
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
        max_length=eval_max_input_seq_len,
        memory_length=rmt_train_args.memory_length,
        memory_position=rmt_train_args.memory_position,
        max_session=args.test_max_session,
        mode='eval' if not args.load_baseline else 'baseline',
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
    # if args.use_lora:
    #     old_state_dict = model.state_dict
    #     # TODO: change all checkpoint names
    #     # NOTE: I have commented out
    #     # `to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}`
    #     # from peft.get_peft_model_state_dict.
    #     model.state_dict = (
    #         lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    #             model, type(model))

    if not args.test_only:
        trainer.train()
        # model.state_dict = old_state_dict
        model.save_pretrained(f"{rmt_train_args.output_dir}")
    trainer.evaluate(test_dataset, metric_key_prefix="test")
    wandb.finish()


if __name__ == "__main__":
    main()
