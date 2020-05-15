import random
from datetime import datetime

import numpy as np

from models.embedding import OneHotEmbedding
from utils.logistic import softmax
from utils.plot import plot_logistic_bigram, plot_smoothed_losses, plot_losses


class FeedForward():
    def __init__(self, embedding: OneHotEmbedding):
        self.embedding = embedding

        self._init_params()

    def _init_params(self):
        self.in_dim = self.embedding.D
        self.out_dim = self.embedding.D

        self.W = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)

    def _features_labels(self, sentence, n_gram):
        sentence = [self.embedding.start_idx] + sentence + [self.embedding.end_idx]

        sep = n_gram - 1
        n = len(sentence) - sep

        features = np.zeros((n, self.in_dim))
        labels = np.zeros((n, self.out_dim))

        features[np.arange(n), sentence[:n]] = 1
        labels[np.arange(n), sentence[sep:]] = 1

        return features, labels

    def _forward(self, features):
        return softmax(features.dot(self.W))

    def _backward(self, features, labels, preds, lr):
        self.W -= lr * features.T.dot(preds - labels)

    def _step(self, preds, labels, losses):
        loss = -np.sum(labels * np.log(preds)) / len(preds) - 1
        losses.append(loss)
        return loss

    def _bigram_forward(self, features, labels, bigram_losses):
        bigram_predictions = softmax(features.dot(np.log(self.embedding.bigram_probs)))
        bigram_loss = -np.sum(labels * np.log(bigram_predictions)) / len(bigram_predictions)
        bigram_losses.append(bigram_loss)

    def train(self, epochs=1, lr=1e-1, n_gram=2):

        t0 = datetime.now()

        bigram_losses = []
        losses = []

        for epoch in range(epochs):

            random.shuffle(self.embedding.embed_sentences)

            for idx, sentence in enumerate(self.embedding.embed_sentences):

                features, labels = self._features_labels(sentence, n_gram)

                preds = self._forward(features)

                self._backward(features, labels, preds, lr)

                loss = self._step(preds, labels, losses)

                self._bigram_forward(features, labels, bigram_losses)

                if idx % 10 == 0:
                    print(f'epoch:{epoch} sentence: {idx}/{len(self.embedding.embed_sentences)} loss: {loss}')

        print(f'Training Time: {datetime.now() - t0}')

        plot_losses(losses)
        plot_smoothed_losses(bigram_losses, losses)
        plot_logistic_bigram(self.embedding.bigram_probs, W=self.W)


class FeedForward_Hidden():
    def __init__(self, embedding: OneHotEmbedding, hidden_dim):
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self._init_params()

    def _init_params(self):
        self.vocab_dim = self.embedding.D

        self.W_1 = np.random.randn(self.vocab_dim, self.hidden_dim) / np.sqrt(self.vocab_dim)
        self.W_2 = np.random.randn(self.hidden_dim, self.vocab_dim) / np.sqrt(self.hidden_dim)

    def _features_labels(self, sentence, n_gram):
        sentence = [self.embedding.start_idx] + sentence + [self.embedding.end_idx]

        sep = n_gram - 1
        n = len(sentence) - sep

        features = np.zeros((n, self.vocab_dim))
        labels = np.zeros((n, self.vocab_dim))

        features[np.arange(n), sentence[:n]] = 1
        labels[np.arange(n), sentence[sep:]] = 1

        return features, labels

    def _forward(self, features):
        hidden = np.tanh(features.dot(self.W_1))
        preds = softmax(hidden.dot(self.W_2))
        return hidden, preds

    def _backward(self, features, labels, hidden, preds, lr):
        self.W_2 -= lr * hidden.T.dot(preds - labels)
        dhidden = (preds - labels).dot(self.W_2.T) * (1 - hidden * hidden)
        self.W_1 -= lr * features.T.dot(dhidden)

    def _step(self, preds, labels, losses):
        loss = -np.sum(labels * np.log(preds)) / len(preds) - 1
        losses.append(loss)
        return loss

    def _bigram_forward(self, features, labels, bigram_losses):
        bigram_predictions = softmax(features.dot(np.log(self.embedding.bigram_probs)))
        bigram_loss = -np.sum(labels * np.log(bigram_predictions)) / len(bigram_predictions)
        bigram_losses.append(bigram_loss)

    def train(self, epochs=1, lr=1e-2, n_gram=2):

        t0 = datetime.now()

        bigram_losses = []
        losses = []

        for epoch in range(epochs):

            random.shuffle(self.embedding.embed_sentences)

            for idx, sentence in enumerate(self.embedding.embed_sentences):

                features, labels = self._features_labels(sentence, n_gram)

                hidden, preds = self._forward(features)

                self._backward(features, labels, hidden, preds, lr)

                loss = self._step(preds, labels, losses)

                self._bigram_forward(features, labels, bigram_losses)

                if idx % 10 == 0:
                    print(f'epoch:{epoch} sentence: {idx}/{len(self.embedding.embed_sentences)} loss: {loss}')

        print(f'Training Time: {datetime.now() - t0}')

        plot_losses(losses)
        plot_smoothed_losses(bigram_losses, losses)
        plot_logistic_bigram(self.embedding.bigram_probs, W_1W_2=(self.W_1, self.W_2))
