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


def plot_logistic_bigram(bigram_probs,W=None,W_1W_2:tuple=None):
    plt.subplot(1, 2, 1)

    weight = W or W_1W_2
    if isinstance(weight,tuple):
        W_1,W_2 = weight
        plt.title("Neural Network Model")
        plt.imshow(np.tanh(W_1).dot(W_2))
    else:
        plt.title("Logistic Model")
        plt.imshow(softmax(weight))
    plt.subplot(1, 2, 2)
    plt.title("Bigram Probs")
    plt.imshow(bigram_probs)
    plt.show()
