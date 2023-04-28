import math

import torch
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

ENCODER_DECODER_ARCH_NAMES = ['t5', 't0', 'bart', 'blenderbot']
MEM_TOKEN = '[mem_{}]'


def load_transformer_LM_tokenizer(model_name_or_path, tokenizer_name_or_path=None, **kwargs):
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if any(name in model_name_or_path.lower() for name in ENCODER_DECODER_ARCH_NAMES):
        model = RMTForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)
    else:
        raise NotImplementedError
        # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        # # open-ended generation
        # tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def prepare_rmt_model(model, tokenizer, memory_length, memory_position, write_memory_position=None):
    tokenizer.add_special_tokens(
        {'additional_special_tokens': [MEM_TOKEN.format(i) for i in range(memory_length)]})
    model.resize_token_embeddings(len(tokenizer))
    model.config.memory_length = memory_length
    model.config.memory_position = memory_position
    model.config.write_memory_position = write_memory_position if write_memory_position else memory_position


class RMTForSeq2SeqLM(BlenderbotForConditionalGeneration):
    # TODO: make it AutoModel-based later. Now, just override BlenderBot
    def forward(self,
                input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                return_dict=True,
                **kwargs):
        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        else:
            raise ValueError("one of input_ids or inputs_embeds must not be None")

        if seq_len <= self.config.max_position_embeddings:
            # if not too long, just normal forward will do.
            return super().forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   inputs_embeds=inputs_embeds,
                                   **kwargs)

        # seq_len too long; recurrent encoder forward
        input_tensor = input_ids if input_ids is not None else inputs_embeds
        input_tensor, memory_tensor = self.split_memory(input_tensor, self.config.memory_position,
                                                        self.config.memory_length)
        input_segments = torch.split(input_tensor,
                                     self.config.max_position_embeddings -
                                     self.config.memory_length,
                                     dim=1)
        if attention_mask is not None:
            attention_mask, memory_mask = self.split_memory(attention_mask,
                                                            self.config.memory_position,
                                                            self.config.memory_length)
            attention_mask_segments = torch.split(attention_mask,
                                                  self.config.max_position_embeddings -
                                                  self.config.memory_length,
                                                  dim=1)
        prev_memory = None
        for i, segment in enumerate(input_segments):
            # for each segment, recurrently pass encoder
            segment_embeds, memory_embeds = self.append_memory(segment,
                                                               memory_tensor,
                                                               prev_memory=prev_memory)
            attention_mask_seg = None
            if attention_mask is not None:
                attention_mask_seg = attention_mask_segments[i]
                attention_mask_seg = self.append_mem_tensor(attention_mask_seg, memory_mask)

            encoder_outputs = self.get_encoder()(inputs_embeds=segment_embeds,
                                                 attention_mask=attention_mask_seg,
                                                 return_dict=True,
                                                 output_hidden_states=True)
            last_hidden_state = encoder_outputs.last_hidden_state  # [B, T, E]
            _, memory_embeds = self.split_memory(last_hidden_state,
                                                 self.config.write_memory_position,
                                                 self.config.memory_length)
            memory_tensor = memory_embeds
            prev_memory = memory_embeds
            # TODO: use .detach() here well for Truncated BPTT.

        return super().forward(encoder_outputs=encoder_outputs,
                               attention_mask=attention_mask_seg,
                               return_dict=True,
                               **kwargs)

    def split_memory(self, tensor, mem_pos, mem_len):
        if mem_pos == 'left':
            memory_tensor = tensor[:, :mem_len]
            input_tensor = tensor[:, mem_len:]
        elif mem_pos == 'right':
            memory_tensor = tensor[:, -mem_len:]
            input_tensor = tensor[:, :-mem_len]
        else:  # split
            left, right = math.floor(mem_len / 2), math.ceil(mem_len / 2)
            memory_tensor = (tensor[:, :left], tensor[:, -right:])
            input_tensor = tensor[:, left:-right]

        return input_tensor, memory_tensor

    def append_memory(self, input_tensor, memory_tensor, prev_memory=None):
        if input_tensor.dtype == torch.long:
            inputs_embeds = self.get_input_embeddings()(input_tensor)
        else:
            inputs_embeds = input_tensor
        if isinstance(memory_tensor, tuple):
            if memory_tensor[0].dtype == torch.long:
                memory_embeds = (self.get_input_embeddings()(mem) for mem in memory_tensor)
            else:
                memory_embeds = memory_tensor
        else:
            if memory_tensor.dtype == torch.long:
                memory_embeds = self.get_input_embeddings()(memory_tensor)
            else:
                memory_embeds = memory_tensor

        if prev_memory is not None:
            memory_embeds = self.memory_gate(memory_embeds, prev_memory)

        inputs_embeds = self.append_mem_tensor(inputs_embeds, memory_embeds)

        return inputs_embeds, memory_embeds

    def append_mem_tensor(self, input_tensor, memory_tensor):
        if self.config.memory_position == 'left':
            input_tensor = torch.cat([memory_tensor, input_tensor], dim=1)
        elif self.config.memory_position == 'right':
            input_tensor = torch.cat([memory_tensor, input_tensor], dim=1)
        else:
            input_tensor = torch.cat([memory_tensor[0], input_tensor, memory_tensor[1]], dim=1)
        return input_tensor

    def memory_gate(self, memory_embeds, prev_memory):
        # TODO: try different ideas
        return memory_embeds
