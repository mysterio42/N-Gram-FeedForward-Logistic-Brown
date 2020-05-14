import numpy as np
from matplotlib import pyplot as plt

from utils.logistic import smoothed_loss
from utils.logistic import softmax


def plot_losses(losses):
    plt.plot(losses)


def plot_smoothed_losses(bigram_losses, losses):
    avg_bigram_loss = np.mean(bigram_losses)
    print("avg_bigram_loss:", avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')

    plt.plot(smoothed_loss(losses))
    plt.show()


def plot_logistic_bigram(W_1,W_2, bigram_probs):
    plt.subplot(1, 2, 1)
    plt.title("Neural Network Model")
    plt.imshow(np.tanh(W_1).dot(W_2))
    plt.subplot(1, 2, 2)
    plt.title("Bigram Probs")
    plt.imshow(bigram_probs)
    plt.show()
