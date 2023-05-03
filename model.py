"""
Our RMT is an encoder-decoder model based on BlenderBot.
"""
import math

import torch
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration
from transformers.models.blenderbot.configuration_blenderbot import BlenderbotConfig
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention

ENCODER_DECODER_ARCH_NAMES = ['t5', 't0', 'bart', 'blenderbot']
MEM_TOKEN = '[mem_{}]'


def load_transformer_LM_tokenizer(model_name_or_path, tokenizer_name_or_path=None, **kwargs):
    add_speaker_tokens = kwargs.pop('add_speaker_tokens', False)
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
    memory_length = kwargs.get('memory_length', None)
    if memory_length is not None and memory_length > 0:
        special_tokens = [MEM_TOKEN.format(i) for i in range(memory_length)]
        if add_speaker_tokens:
            special_tokens += ['[Speaker 1]', '[Speaker 2]']
        tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        model.config.memory_length = memory_length

    return model, tokenizer


class RMTForSeq2SeqLM(BlenderbotForConditionalGeneration):
    # TODO: make it AutoModel-based later. Now, just override BlenderBot
    def __init__(self,
                 config: BlenderbotConfig,
                 memory_length=10,
                 memory_position='left',
                 write_memory_position='right',
                 memory_gate_type='none'):
        super().__init__(config)
        self.config.memory_length = memory_length
        self.config.memory_position = memory_position
        self.config.write_memory_position = write_memory_position
        self.config.memory_gate_type = memory_gate_type

        if self.config.memory_length > 0:
            self.memory_proj = torch.nn.Linear(config.d_model, config.d_model)
        if self.config.memory_gate_type == 'attention':
            self.memory_attention = BlenderbotAttention(embed_dim=config.d_model,
                                                        num_heads=config.encoder_attention_heads,
                                                        dropout=config.attention_dropout)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                session_ids=None,
                return_dict=True,
                return_encoder_outputs_only=False,
                **kwargs):
        del session_ids  # unused

        encoder_outputs = kwargs.get('encoder_outputs', None)
        if encoder_outputs is not None:
            return super().forward(attention_mask=attention_mask,
                                   return_dict=return_dict,
                                   **kwargs)

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        else:
            raise ValueError("one of input_ids or inputs_embeds must not be None")

        if seq_len <= self.config.max_position_embeddings or self.config.memory_length < 1:
            # if not too long, just normal forward will do.
            # Also, if memory_length == 0, this is just blenderbot.
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
            _, new_memory_embeds = self.split_memory(last_hidden_state,
                                                     self.config.write_memory_position,
                                                     self.config.memory_length)
            memory_tensor = new_memory_embeds
            prev_memory = memory_embeds
            # TODO: use .detach() here well for Truncated BPTT.

        if return_encoder_outputs_only:
            return encoder_outputs, attention_mask_seg

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
            inputs_embeds = self.get_input_embeddings()(input_tensor) * self.get_encoder().embed_scale
        else:
            inputs_embeds = input_tensor
        if isinstance(memory_tensor, tuple):
            if memory_tensor[0].dtype == torch.long:
                memory_embeds = (self.get_input_embeddings()(mem) * self.get_encoder().embed_scale for mem in memory_tensor)
            else:
                memory_embeds = memory_tensor
        else:
            if memory_tensor.dtype == torch.long:
                memory_embeds = self.get_input_embeddings()(memory_tensor) * self.get_encoder().embed_scale
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
            input_tensor = torch.cat([input_tensor, memory_tensor], dim=1)
        else:
            input_tensor = torch.cat([memory_tensor[0], input_tensor, memory_tensor[1]], dim=1)
        return input_tensor

    def memory_gate(self, memory_embeds, prev_memory):
        if self.config.memory_gate_type == 'none':
            return memory_embeds

        if self.config.memory_gate_type == 'residual':
            return memory_embeds + prev_memory

        if self.config.memory_gate_type == 'attention':
            # prev_memory  [B, T, E]
            # memory_embeds  [B, T, E]
            memory, _, _ = self.memory_attention(
                hidden_states=memory_embeds,
                key_value_states=prev_memory,
            )
            return memory

        if self.config.memory_gate_type == 'zero_conv':
            raise NotImplementedError
