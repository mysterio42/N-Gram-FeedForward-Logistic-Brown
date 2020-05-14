import operator
from collections import Counter

import nltk
from nltk.corpus import brown

from utils.onehot import build_bigram_probs


class OneHotEmbedding:
    def __init__(self, sep, smoothing):
        self.word2idx = {}
        self.idx2word = {}
        self.start_idx = ''
        self.end_idx = ''
        self.D = 0
        self.embed_sentences = []
        self.smoothing = smoothing
        self.sep = sep

    def build(self):
        self._encode_tokens()
        self._encode_sentences()
        self._decode_sentences()
        self._build_bigrams(self.smoothing)

    def _encode_tokens(self):
        nltk.download('brown')

        for idx, token in enumerate(['START', 'END']):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        most_common_words = sorted(Counter(brown.words()).items(), key=operator.itemgetter(1), reverse=True)[:self.sep]

        for idx, (token, _) in ((idx, (token, _))
                                for idx, (token, _) in enumerate(most_common_words)):
            self.word2idx[token] = idx + 2
            self.idx2word[idx + 2] = token

        self.start_idx, self.end_idx = self.word2idx['START'], self.word2idx['END']

        self.D = len(self.word2idx)

    def _encode_sentences(self):
        self.embed_sentences = [emb_sen for emb_sen in (([self.word2idx[token.lower()]
                                                          for token in sentence
                                                          if token.lower() in self.word2idx])
                                                        for sentence in brown.sents())
                                if len(emb_sen) > 0]

        assert all([True if len(el) > 0 else False for el in
                    self.embed_sentences]) == True, " some embedded sentence doesn't exist "

    def _decode_sentences(self):
        self.decode_sents = [[self.idx2word[idx] for idx in sentence if idx in self.idx2word]
                             for sentence in self.embed_sentences]

    def _build_bigrams(self, smoothing):
        self.bigram_probs = build_bigram_probs(self.embed_sentences,
                                               self.D,
                                               self.start_idx, self.end_idx,
                                               smoothing)
