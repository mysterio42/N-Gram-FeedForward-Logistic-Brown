import numpy as np

from models.embedding import OneHotEmbedding
from models.net import FeedForward,FeedForward_Hidden

np.set_printoptions(edgeitems=50)
np.random.seed(1)

if __name__ == '__main__':
    emb = OneHotEmbedding(200, 0.1)
    emb.build()

    # model = FeedForward(embedding=emb)
    model = FeedForward_Hidden(embedding=emb,hidden_dim=100)
    model.train(epochs=1, n_gram=2)
