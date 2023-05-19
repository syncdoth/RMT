from dataclasses import dataclass, field
import json
import time

from transformers import DataCollatorForSeq2Seq
from transformers.hf_argparser import HfArgumentParser
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import MscDataset
from trainer import RMTTrainingArgs
from model import load_transformer_LM_tokenizer
from eval_metric import Metrics

@dataclass
class ExperimentArgs:
    # facebook/blenderbot-3B
    model_name: str = field(default='facebook/blenderbot-400M-distill')
    tokenizer_name: str = field(default=None, metadata={"help": "Will default to --model_name."})
    # data
    test_data_path: str = 'msc/session_5/test.txt'
    test_max_session: int = 5
    test_target_session: int = None

    load_baseline: bool = False
    load_checkpoint: str = field(default=None)
    load_peft_checkpoint: str = field(default=None)
    train_8bit: bool = False
    out_file: str = 'out.jsonl'

    add_speaker_tokens: bool = False

    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    task: str = 'default'

    sample_idx : bool  = False


@torch.inference_mode()
def infer_testset(model, tokenizer, test_dataloader, generate_kwargs, device, fout):
    for batch in tqdm(test_dataloader):
        seqlen = batch['input_ids'].shape[1]
        labels = batch.pop('labels')
        labels[labels == -100] = tokenizer.eos_token_id
        session_ids = batch.pop('session_ids')
        session_ids = session_ids.squeeze(-1).tolist()

        # device
        batch = {k: v.to(device) for k, v in batch.items()}
        if seqlen <= model.config.max_position_embeddings:
            generated = model.generate(**batch, **generate_kwargs)
        else:
            encoder_outputs, attention_mask = model(**batch, return_encoder_outputs_only=True)
            generated = model.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask,
                                       **generate_kwargs)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
        answer = tokenizer.batch_decode(labels, skip_special_tokens=False)
        inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)

        for i, p, t, s in zip(inputs, decoded, answer, session_ids):
            i = i.replace(tokenizer.pad_token, '')
            p = p.replace(tokenizer.pad_token, '')
            t = t.replace(tokenizer.pad_token, '')
            t = t.replace(tokenizer.eos_token, '')
            line = {'input': i, 'pred': p, 'target': t, 'session': s}
            json.dump(line, fout)
            fout.write('\n')


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

    model, tokenizer = load_transformer_LM_tokenizer(
        args.model_name,
        tokenizer_name_or_path=args.tokenizer_name,
        memory_length=rmt_train_args.memory_length,
        memory_position=rmt_train_args.memory_position,
        write_memory_position=rmt_train_args.write_memory_position,
        memory_gate_type=rmt_train_args.memory_gate_type,
        load_in_8bit=args.train_8bit,
        device_map=device_map,
        add_speaker_tokens=args.add_speaker_tokens,
    )

    if args.load_peft_checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)

    elif args.use_lora:
        from peft import (get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType,
                          prepare_model_for_int8_training, PeftModel)
        modules_to_save = None
        if rmt_train_args.memory_length > 0:
            modules_to_save = ['shared', 'memory_proj']  # save embedding
        if rmt_train_args.memory_gate_type == 'attention':
            modules_to_save.append('memory_attention')
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules='model.*(q_proj|k_proj|v_proj|o_proj)$',
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, peft_config)

    if args.load_checkpoint:
        start = time.time()
        print("load full checkpoint")
        model.load_state_dict(torch.load(args.load_checkpoint), strict=False)
        print(f"done in {time.time() - start:.2f}s")


    # for eval, defaults to use the entire sequence.
    if rmt_train_args.eval_num_segments > 0:
        eval_max_input_seq_len = (model.config.max_position_embeddings -
                                  rmt_train_args.memory_length) * rmt_train_args.eval_num_segments
        print(f"During eval, {rmt_train_args.eval_num_segments} segments will be used.",
              f"This leads to {eval_max_input_seq_len - 1} tokens in encoder input.")
    else:
        print("During eval, entire sequence (history + query) will be used, no matter how long.")
        eval_max_input_seq_len = -1

    test_dataset = MscDataset(
        tokenizer,
        args.test_data_path,
        max_length=eval_max_input_seq_len,
        memory_length=rmt_train_args.memory_length,
        memory_position=rmt_train_args.memory_position,
        max_session=args.test_max_session,
        mode='eval' if not args.load_baseline else 'baseline',
        target_session=args.test_target_session,
        task=args.task,
    )
    if args.sample_idx:
        # QUICK SAVE OF INDICES FOR HUMAN EVAL
        import random
        idx = random.sample(range(len(test_dataset)), 100)
        with open(f'generated/final/human_eval_session{args.test_target_session}_idx.txt', 'w') as f:
            for i in idx:
                f.write(str(i) + '\n')
            exit()
    with open(f'generated/final/human_eval_session{args.test_target_session}_idx.txt', 'r') as f:
        indices = [int(x.strip()) for x in f.readlines()]

    test_dataloader = DataLoader(test_dataset, batch_size=rmt_train_args.per_device_eval_batch_size, shuffle=False,
                                 sampler=indices,
                                 collate_fn=DataCollatorForSeq2Seq(tokenizer))

    generate_kwargs = dict(
        max_new_tokens=40,
        do_sample=True,
        top_p=0.95,
        # num_beams=8,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    with open(args.out_file, 'w') as fout:
        infer_testset(model, tokenizer, test_dataloader, generate_kwargs, device, fout)

    # eval_metric
    metrics = Metrics(args.out_file)
    dist1, dist2, dist3, dist4 = metrics.distinct_metrics()
    bleu1, bleu2, bleu3, bleu4 = metrics.bleu_metrics()
    metrics_dial = {
        "dist1": dist1, "dist2": dist2, "dist3": dist3, "dist4":dist4,
        "bleu1":bleu1, "bleu2":bleu2, "bleu3":bleu3, "bleu4":bleu4,
    }
    print(metrics_dial)

if __name__ == "__main__":
    main()
