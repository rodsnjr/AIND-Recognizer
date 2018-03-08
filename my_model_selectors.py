import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def __train_score(self, num_states=1):
        hmm_model = self.base_model(num_states)
        log_likelihood = hmm_model.score(self.X, self.lengths)

        return hmm_model, log_likelihood
        
    def score_bic(self, num_states, log_likelihood):
        num_data_points = sum(self.lengths)
        num_free_params = ( num_states ** 2 ) + ( 2 * num_states * num_data_points ) - 1
        return (-2 * log_likelihood) + (num_free_params * np.log(num_data_points))

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        scores, models = [], []

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model, log_likelihood = self.__train_score(num_states)
                scores.append(self.score_bic(num_states, log_likelihood))
                models.append(hmm_model)
            except Exception as e:
                pass
        # Avoid any possible errors
        assert len(scores) == len(models)

        if len(scores) > 2:
            best_model = models[np.argmin(scores)]
            return best_model
        return trained_models.pop()

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def __train_score(self, num_states=1):
        hmm_model = self.base_model(num_states)
        log_likelihood = hmm_model.score(self.X, self.lengths)

        return hmm_model, log_likelihood

    def get_other_words(self):
        other_words = []
        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        return other_words
    
    def log_lh_ow(self, model, other_words):
        # Calculate the log for other words
        return [model.score(word[0], word[1]) for word in other_words]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        other_words = self.get_other_words()
        models, score_dics = [], []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model, log_lh = self.__train_score(num_states)
                score_dic = log_lh - np.mean(self.log_lh_ow(model, other_words))
                score_dics.append(score_dic)
                models.append(model)
            except Exception as e:
                pass
        
        assert len(models) == len(score_dics)
        
        if len(score_dics) > 2:
            best_model = models[np.argmin(score_dics)]
            return best_model
        return trained_models.pop()

class SelectorCV(ModelSelector):
    ''' 
    select best model based on average log Likelihood of cross-validation folds
    '''
    def __train_score(self, num_states=1):
        hmm_model = self.base_model(num_states)
        log_likelihood = hmm_model.score(self.X, self.lengths)

        return hmm_model, log_likelihood
    
    def __score_kf(self, num_states):
        # Return a generator for each one of the splits
        for train_index, test_index in self.kf.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_index, self.sequences)
            X_test, lengths_test = combine_sequences(test_index, self.sequences)
            yield self.__train_score(num_states)

    def select(self, k=3):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.kf = KFold(n_splits = k, shuffle = False, random_state = None)
        log_likelihoods = []
        trained_models = []
        
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                if len(self.sequences) > 2: # Use KF only on suficient data
                    # Get all the sequenced generated values
                    for model, log_lh in self.__score_kf(num_states):
                        trained_models.append(model)
                        log_likelihoods.append(log_lh)
                else: # Dont't use KF
                    hmm_model, log_likelihood = self.__train_score(num_states)
                    log_likelihoods.append(log_likelihood)
                    trained_models.append(hmm_model)
            except Exception as e:
                # print('Error in training %s' % e)
                pass
        
        assert len(trained_models) == len(log_likelihoods)
        # Get the means for all the log_likelihoods
        # From the means get the best one
        if len(log_likelihoods) > 2:
            best_one = np.argmax(log_likelihoods)
            # Get the best trained model
            return trained_models[best_one]
        return trained_models.pop()