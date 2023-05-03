from itertools import chain

from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pandas as pd

from model import MEM_TOKEN


class MscDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=-1,
        memory_length=8,
        memory_position='left',
        max_session=1,
        mode='train',
        target_session=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.memory_length = memory_length
        self.memory_position = memory_position
        self.max_session = max_session
        self.mode = mode  # [train, eval]
        self.target_session = target_session
        # self.identity = 'Speaker 2'  # TODO

        self.data, self.data_stats = self.load_data(data_path)
        self.histories, self.queries, self.responses, self.session_ids, self.data_idx = self.format_data()
        self.memory_tokens = self.tokenizer.encode(''.join(
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
                    f"[Speaker {u % 2 + 1}]: {normalize_text(dialog['text'])}"
                    for u, dialog in enumerate(session['dialog'])
                ]
                history.append(session_history)
                num_dialogs.append(len(session_history))

            current_session = [f"[{dialog['id']}]: {normalize_text(dialog['text'])}" for dialog in row['dialog']]
            history.append(current_session)  # list(list(str))
            num_dialogs.append(len(current_session))

            chats.append(history)
            num_dialogs_per_session.append(num_dialogs)
            num_sessions_per_chat.append(len(history))

        data_stats = dict(num_chats=num_chats,
                          num_sessions_per_chat=num_sessions_per_chat,
                          num_dialogs_per_session=num_dialogs_per_session)
        return chats, data_stats

    def format_data(self):
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

        if self.target_session is not None:
            data_idx = []
            for i, sid in enumerate(session_ids):
                if sid + 1 == self.target_session:
                    data_idx.append(i)
        else:
            data_idx = None

        return histories, queries, responses, session_ids, data_idx

    def __len__(self):
        if self.data_idx is not None:
            return len(self.data_idx)
        return len(self.queries)

    def __getitem__(self, idx):
        if self.data_idx is not None:
            idx = self.data_idx[idx]
        history = self.histories[idx]
        query = self.queries[idx]
        response = self.responses[idx]
        sess_ids = self.session_ids[idx]
        if history is None:
            input_ids = query
        else:
            input_ids = list(chain.from_iterable(history)) + query
            if self.mode == 'train':
                # for training, constraint input seq len to
                # (model's max_seq_len - memory_len) * num_segments (-1 to add eos later)
                input_ids = input_ids[-(self.max_length - 1):]
            elif self.mode == 'baseline':
                # for baseline, constrain input seq len to
                # model's max_seq_len (-1 to add eos later)
                input_ids = input_ids[-(self.tokenizer.model_max_length - 1):]
            else:  # eval
                if self.max_length > 0:
                    # this means that we are doing ablation study of number of
                    # segments during eval
                    input_ids = input_ids[-(self.max_length - 1):]

        if self.mode != 'baseline':
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


def normalize_text(text: str):
    text = text.strip()
    text = ' '.join(text.split())
    return text
