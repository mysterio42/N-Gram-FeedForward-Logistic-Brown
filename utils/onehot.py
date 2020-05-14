import numpy as np




def build_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                bigram_probs[start_idx, sentence[i]] += 1
            elif i == len(sentence) - 1:
                bigram_probs[sentence[i], end_idx] += 1
            else:
                bigram_probs[sentence[i - 1], sentence[i]] += 1
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)

    return bigram_probs
