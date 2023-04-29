from itertools import chain

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pandas as pd

from model import MEM_TOKEN


class MscDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=128,
        memory_length=8,
        memory_position='left',
        max_session=1,
        mode='train',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.memory_length = memory_length
        self.memory_position = memory_position
        self.max_session = max_session
        self.mode = mode  # [train, eval]
        # self.identity = 'Speaker 2'  # TODO

        self.data, self.data_stats = self.load_data(data_path)
        if self.mode == "train":
            self.histories, self.queries, self.responses, self.session_ids = self.format_train_data()
        else:
            self.histories, self.queries, self.responses, self.session_ids = self.format_eval_data()
        self.memory_tokens = self.tokenizer.encode(' '.join(
            [MEM_TOKEN.format(i) for i in range(memory_length)]))[:-1]

    def load_data(self, path):
        # columns: ['personas', 'dialog', 'metadata', 'previous_dialogs', 'init_personas']
        df = pd.read_json(path, lines=True)
        chats = []
        num_chats = len(df)
        num_sessions_per_chat = []
        num_dialogs_per_session = []
        for i, row in tqdm(df.iterrows(), desc='load data', total=len(df)):
            history = []
            num_dialogs = []
            for session in row['previous_dialogs']:
                # session: {'personas': list, 'dialog': list}
                session_history = [
                    f"[Speaker {u % 2 + 1}]: {dialog['text']}"
                    for u, dialog in enumerate(session['dialog'])
                ]
                history.append(session_history)
                num_dialogs.append(len(session_history))

            current_session = [f"[{dialog['id']}]: {dialog['text']}" for dialog in row['dialog']]
            history.append(current_session)  # list(list(str))
            num_dialogs.append(len(current_session))

            chats.append(history)
            num_dialogs_per_session.append(num_dialogs)
            num_sessions_per_chat.append(len(history))

        data_stats = dict(num_chats=num_chats,
                          num_sessions_per_chat=num_sessions_per_chat,
                          num_dialogs_per_session=num_dialogs_per_session)
        return chats, data_stats

    def format_train_data(self):
        histories = []
        queries = []
        responses = []
        session_ids = []
        for chat in tqdm(self.data, desc='format data'):
            # TODO: sequence per chat, not session
            curr_seqlen = 0
            sequence = []
            for sess_id, session in enumerate(chat[:self.max_session]):
                for dialog in session:
                    encoded = self.tokenizer.encode(dialog)[:-1]  # skip eos
                    if curr_seqlen + len(encoded) + 1 <= self.max_length:
                        sequence.append(encoded)
                        curr_seqlen += len(encoded)
                    else:
                        history = sequence[:-1]
                        query = sequence[-1]
                        response = encoded
                        if not history:
                            history = None
                        histories.append(history)
                        queries.append(query)
                        responses.append(response[:self.tokenizer.model_max_length - 1])
                        session_ids.append(sess_id)

                        # encoded text as next
                        if history is not None:
                            oldest = history.pop(0)
                            sequence = history + [query, response]
                        else:
                            oldest = []
                            sequence = [query, response]
                        curr_seqlen = curr_seqlen - len(oldest) + len(query) + len(response)

        return histories, queries, responses, session_ids

    def format_eval_data(self):
        histories = []
        queries = []
        responses = []
        session_ids = []
        for chat in tqdm(self.data, desc='format data'):
            sequence = []
            for sess_id, session in enumerate(chat[:self.max_session]):
                for dialog in session:
                    # TODO: just all sentence (not by length)
                    encoded = self.tokenizer.encode(dialog)[:-1]  # skip eos
                    sequence.append(encoded)
                    if len(sequence) == 1:
                        continue
                    elif len(sequence) == 2:
                        histories.append(None)
                        queries.append(sequence[-2])
                        response = sequence[-1]
                        response = response[:self.tokenizer.model_max_length - 1]
                        responses.append(response)
                    else:
                        histories.append(sequence[:-2])
                        queries.append(sequence[-2])
                        response = sequence[-1]
                        response = response[:self.tokenizer.model_max_length - 1]
                        responses.append(response)
                    session_ids.append(sess_id)
        return histories, queries, responses, session_ids

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        history = self.histories[idx]
        query = self.queries[idx]
        response = self.responses[idx]
        sess_ids = self.session_ids[idx]
        if history is None:
            input_ids = query
        else:
            input_ids = list(chain.from_iterable(history)) + query
        # add memory tokens
        if self.memory_position == 'left':
            input_ids = self.memory_tokens + input_ids
        elif self.memory_position == 'right':
            input_ids = input_ids + self.memory_tokens
        else:
            half = self.memory_length // 2
            input_ids = self.memory_tokens[:half] + input_ids + self.memory_tokens[half:]
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        labels = response + [self.tokenizer.eos_token_id]

        return dict(input_ids=input_ids, labels=labels, session_ids=[sess_ids])
