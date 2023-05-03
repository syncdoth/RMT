from collections import Counter
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import torch
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
import json

class Metrics():
    # def __init__(self, preds, gt, index2word):
    def __init__(self, output_file):
        self.load_output(output_file)

    def load_output(self, output_file):
        with open(output_file, 'r') as json_file:
            json_list = list(json_file)

        self.preds = []
        self.targets = []
        for json_str in json_list:
            self.preds.append(json.loads(json_str)['pred'])
            self.targets.append(json.loads(json_str)['target'])



    def distinct_metrics(self):
        """ Calculate intra/inter distinct 1/2. """
        intra_dist1, intra_dist2 = [], []
        unigrams_all, bigrams_all, trigrams_all, quadgrams_all = Counter(), Counter(), Counter(), Counter()
        for seq in self.preds:
            unigrams = Counter(seq)
            bigrams = Counter(zip(seq, seq[1:]))
            trigrams = Counter(zip(seq, seq[1:], seq[2:]))
            quadgrams = Counter(zip(seq, seq[1:], seq[2:], seq[3:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))
            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)
            trigrams_all.update(trigrams)
            quadgrams_all.update(quadgrams)
        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        inter_dist3 = (len(trigrams_all)+1e-12) / (sum(trigrams_all.values())+1e-5)
        inter_dist4 = (len(quadgrams_all)+1e-12) / (sum(quadgrams_all.values())+1e-5)
        intra_dist1 = np.average(intra_dist1)
        intra_dist2 = np.average(intra_dist2)
        return inter_dist1, inter_dist2, inter_dist3, inter_dist4

    def bleu_metrics(self):
        bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
        count = 0

        for out, tar in zip(self.preds, self.targets):
            bleu1+=sentence_bleu([tar], out, weights=(1, 0, 0, 0))
            bleu2+=sentence_bleu([tar], out, weights=(0, 1, 0, 0))
            bleu3+=sentence_bleu([tar], out, weights=(0, 0, 1, 0))
            bleu4+=sentence_bleu([tar], out, weights=(0, 0, 0, 1))
            count+=1

        return bleu1/count, bleu2/count, bleu3/count, bleu4/count