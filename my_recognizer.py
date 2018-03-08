import warnings
from asl_data import SinglesData
import numpy as np

def convert_dict(likelihoods, words):
    words_likelihoods = {}
    for word, likelihood in zip(words, likelihoods):
        words_likelihoods[word] = likelihood
    return words_likelihoods

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_ind in range(0, len(test_set.get_all_Xlengths())):
        # data from the current word
        feature_lists_seq, seq_length = test_set.get_item_Xlengths(word_ind)
        likelihoods = []
        words = []

        for word, model in models.items():
            try:
                score = model.score(feature_lists_seq, seq_length)
                likelihoods.append(score)
                words.append(word)
            except:
                # Set a value to avoid this model
                likelihoods.append(float('-inf'))
                words.append(word)
        best_guess = np.argmax(likelihoods)
        probabilities.append(convert_dict(likelihoods, words))
        guesses.append(words[best_guess])
    
    return probabilities, guesses