from itertools import chain
import random

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
        task='default',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.memory_length = memory_length
        self.memory_position = memory_position
        self.max_session = max_session
        self.mode = mode  # [train, eval]
        self.target_session = target_session
        self.identity = None  # TODO
        self.sep_token = '</s> <s>'
        self.task = task

        self.data, self.data_stats = self.load_data(data_path)
        self.histories, self.queries, self.responses, self.session_ids, self.data_idx = self.format_data()
        # NOTE: this used to be ' '.join(). changed it to ''.join()
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
                    # f"[Speaker {u % 2 + 1}]: {normalize_text(dialog['text'])}"
                    normalize_text(dialog['text'])
                    for u, dialog in enumerate(session['dialog'])
                ]
                history.append(session_history)
                num_dialogs.append(len(session_history))

            # current_session = [f"[{dialog['id']}]: {normalize_text(dialog['text'])}" for dialog in row['dialog']]
            current_session = [normalize_text(dialog['text']) for dialog in row['dialog']]
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
            current_chat_idx = len(histories)
            sequence = []
            for sess_id, session in enumerate(chat[:self.max_session]):
                for dialog in session:
                    # TODO: just all sentence (not by length)
                    # encoded = self.tokenizer.encode(dialog)[:-1]  # skip eos

                    sequence.append(dialog)
                    if len(sequence) == 1:
                        continue
                    elif len(sequence) == 2:
                        histories.append(None)
                        queries.append(sequence[-2])
                        response = sequence[-1]
                        # response = response[:self.tokenizer.model_max_length - 1]
                        responses.append(response)
                    else:
                        histories.append(sequence[:-2])
                        queries.append(sequence[-2])
                        response = sequence[-1]
                        # response = response[:self.tokenizer.model_max_length - 1]
                        responses.append(response)
                    session_ids.append(sess_id)
            if self.task == 'remember_sess1':
                # in this task, after all is finished,
                idx = random.randint(0, 5)
                idx = idx * 2  # NOTE: the new question should be from Speaker 1, so even idx
                utt_id = current_chat_idx + idx
                histories.append(histories[-1])
                queries.append(queries[utt_id])
                responses.append(responses[utt_id])
                session_ids.append(100)  # unique: session_id 100 is for remember task.

        if self.target_session is not None:
            data_idx = []
            for i, sid in enumerate(session_ids):
                if sid + 1 == self.target_session:
                    data_idx.append(i)
                elif self.task == 'remember_sess1' and sid == 100:
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
            input_ids = self.tokenizer.encode(query)[:-1]
        else:
            history = self.sep_token.join(history)
            input_ids = self.tokenizer.encode(history + '\n' + query)[:-1]
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
            if input_ids[0] == self.tokenizer.bos_token_id:
                input_ids = input_ids[1:]
        # response = response.replace('[Speaker 1]: ', '')
        # response = response.replace('[Speaker 2]: ', '')
        response = self.tokenizer.encode(response)
        labels = response[:self.tokenizer.model_max_length - 1]

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

        return dict(input_ids=input_ids, labels=labels, session_ids=[sess_ids])


def normalize_text(text: str):
    text = text.strip()
    text = ' '.join(text.split())
    return text
