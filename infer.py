from dataclasses import dataclass, field
import json

from transformers import DataCollatorForSeq2Seq
from transformers.hf_argparser import HfArgumentParser
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import MscDataset
from trainer import RMTTrainingArgs
from model import load_transformer_LM_tokenizer


@dataclass
class ExperimentArgs:
    # facebook/blenderbot-3B
    model_name: str = field(default='facebook/blenderbot-400M-distill')
    tokenizer_name: str = field(default=None, metadata={"help": "Will default to --model_name."})
    # data
    test_data_path: str = 'msc/session_5/test.txt'
    test_max_session: int = 5

    load_baseline: bool = False
    load_checkpoint: str = field(default=None)
    load_peft_checkpoint: str = field(default=None)
    train_8bit: bool = False
    out_file: str = 'out.jsonl'



def infer_testset(model, tokenizer, test_dataloader, generate_kwargs, device, fout):
    for batch in tqdm(test_dataloader):
        seqlen = batch['input_ids'].shape[1]
        labels = batch.pop('labels')
        session_ids = batch.pop('session_ids')
        session_ids = session_ids.squeeze(-1).tolist()

        # device
        batch = {k: v.to(device) for k, v in batch.items()}
        if seqlen <= model.config.max_position_embeddings:
            generated = model.generate(**batch, **generate_kwargs)
        else:
            encoder_outputs, inputs_embeds, attention_mask = model(**batch, return_encoder_outputs_only=True)
            generated = model.generate(encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                       **generate_kwargs)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        labels[labels == -100] = tokenizer.eos_token_id
        answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
        inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

        for i, p, t, s in zip(inputs, decoded, answer, session_ids):
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
    )

    if args.load_peft_checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)

    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint), strict=False)

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
    )
    test_dataloader = DataLoader(test_dataset, batch_size=rmt_train_args.per_device_eval_batch_size, shuffle=False,
                                 collate_fn=DataCollatorForSeq2Seq(tokenizer))

    generate_kwargs = dict(
        max_new_tokens=40,
        do_sample=False,
        top_p=1,
        num_beams=8,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    with open(args.out_file, 'w') as fout:
        infer_testset(model, tokenizer, test_dataloader, generate_kwargs, device, fout)


if __name__ == "__main__":
    main()
