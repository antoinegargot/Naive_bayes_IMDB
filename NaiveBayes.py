import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        #Initialization of each instance variables by the type they need.
        self.ALPHA = ALPHA
        self.data = data
        self.vocab_len = 0
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_number_of_reviews = 0
        self.count_positive = defaultdict(int)
        self.count_negative = defaultdict(int)
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.total_positive_term = 0
        self.total_negative_term = 0
        self.logP_positive = defaultdict(int)
        self.logP_negative = defaultdict(int)
        self.deno_pos = 0
        self.deno_neg = 0

        print("Training the model ...")
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #Estimation of the Naive Bayes model parameters

        #Getting the number of unique word in the training corpus
        self.vocab_len = X.shape[1]
        
        #Getting the number of positive and negative reviews based on the label of the training set.
       
        #Positive reviews
        self.num_positive_reviews = np.count_nonzero(Y == 1)
        #Negative reviews
        self.num_negative_reviews = np.count_nonzero(Y == -1)

        #Getting the total number of reviews in the training data set.
        self.total_number_of_reviews = X.shape[0]

        #Count number of word occurences in the corpus for positive and negative reviews and create a dictionnay
        #from it.
        for review in range(self.total_number_of_reviews):
            for word, occurences in zip(X[review].indices, X[review].data):
                if traindata.Y[review] == 1:
                    self.count_positive[word] += occurences
                elif traindata.Y[review] == -1:
                    self.count_negative[word] += occurences

        # Represent the sum of each occurences for each word in class label
        self.total_positive_words = sum(self.count_positive.values())
        self.total_negative_words = sum(self.count_negative.values())
        
        # Represent number of unique words in each label
        self.total_positive_term = len(self.count_positive.keys())
        self.total_negative_term = len(self.count_negative.keys())

        # Denominator number for the computation of P(W|C). C is + or -
        self.deno_pos = self.vocab_len * self.ALPHA + self.total_positive_words
        self.deno_neg = self.vocab_len * self.ALPHA + self.total_negative_words

        # Compute probabilities for each term inside a concept P(W|C) using log in order to avoid
        # future underflow
        for word in range(self.vocab_len):
            self.logP_positive[word] = log((self.count_positive[word] + self.ALPHA) / self.deno_pos)
        for word in range(self.vocab_len):
            self.logP_negative[word] = log((self.count_negative[word] + self.ALPHA) / self.deno_pos)        

    def LogSum(self, logx, logy):
        return max(logx, logy) + log(exp(logx - max(logx, logy)) + exp(logy - max(logx, logy)))


    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        # Naive Bayes Classification
        
        pred_labels = []
        score_word_pos = 1
        score_word_neg = 1
        number_of_training_reviews = X.shape[0]

        for review in range(number_of_training_reviews):

            # Compute the probability of each concept P(C) C is equal to + or - 
            score_word_pos = log(self.num_positive_reviews/ self.total_number_of_reviews)
            score_word_neg = log(self.num_negative_reviews/ self.total_number_of_reviews)

            # Compute the probability that each review belongs to a specific class label 
            for word, occurences in zip(X[review].indices, X[review].data):
                # For positive class
                score_word_pos = score_word_pos + self.logP_positive[word]*occurences
                # For negative class
                score_word_neg = score_word_neg + self.logP_negative[word]*occurences

            # Predict the label of the review.
            if score_word_pos > score_word_neg: 
            # Predict positive
                pred_labels.append(1.0)
            else:
            # Predict negative
                pred_labels.append(-1.0)

        return pred_labels


    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):

        number_positive_reviews = 0
        number_negative_reviews = 0
        predicted_label = 0
        positive_probs = []
        negative_probs = []
        # Compute the number of positives and negatives reviews in the dataset. 
        for review in indexes:
            if test.Y[review] == 1:
                number_positive_reviews += 1
            if test.Y[review] == -1:
                number_negative_reviews += 1

        for review in indexes:

            # Compute the probability of each concept P(C) C is equal to + or - 
            predicted_prob_positive = log(number_positive_reviews/ len(indexes))
            predicted_prob_negative = log(number_negative_reviews/ len(indexes))

            # Compute the probability that each review belongs to a specific class label 
            for word, occurences in zip(test.X[review].indices, test.X[review].data):

                # For positive class
                predicted_prob_positive = predicted_prob_positive + self.logP_positive[word]*occurences

                # For negative class
                predicted_prob_negative = predicted_prob_negative + self.logP_negative[word]*occurences

            final_predicted_prob_positive = exp(predicted_prob_positive - self.LogSum(predicted_prob_positive,predicted_prob_negative))
            final_predicted_prob_negative = exp(predicted_prob_negative - self.LogSum(predicted_prob_positive,predicted_prob_negative))

            positive_probs.append(final_predicted_prob_positive)
            negative_probs.append(final_predicted_prob_negative)

            if final_predicted_prob_positive > final_predicted_prob_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            #print(review, ":" ,test.Y[review], predicted_label, final_predicted_prob_positive, final_predicted_prob_negative)
            #print(test.Y[review], predicted_label, final_predicted_prob_positive, final_predicted_prob_negative, test.X_reviews[review])
        return positive_probs, negative_probs
        #Compute the Precision and Recall based on a threshold

    def Precision_Recall_curve(self, test, positive_probs, negative_probs, indexes):
        positive_precision_recall_points = []
        negative_precision_recall_points = []
        positive_F1 = []
        negative_F1 = []
        for threshold in np.arange(0, 1, 0.01):
            positive_prediction_based_on_threshold = []
            negative_prediction_based_on_threshold = []
            for prob in range(len(indexes)):
                if positive_probs[prob] > threshold :
                    positive_prediction_based_on_threshold.append(1) 
                else :
                    positive_prediction_based_on_threshold.append(-1)
            ev = Eval(positive_prediction_based_on_threshold, test.Y)
            ev.ComputeConfusionMatrix()
            positive_precision_recall_points.append((ev.Recall(), ev.Precision()))
            #positive_F1.append((2*ev.Recall()*ev.Precision()) / (ev.Recall()+ev.Precision()))

            for prob in range(len(indexes)):
                if negative_probs[prob] > threshold :
                    negative_prediction_based_on_threshold.append(-1) 
                else :
                    negative_prediction_based_on_threshold.append(1)
            ev = Eval(negative_prediction_based_on_threshold, test.Y)
            ev.ComputeConfusionMatrix()
            negative_precision_recall_points.append((ev.Recall(), ev.Precision()))
            #negative_F1.append((2*ev.Recall()*ev.Precision()) / (ev.Recall()+ev.Precision()))
        #print(positive_F1, negative_F1)
        #plot each recall and precision points based on threshold
        fig, ax = plt.subplots()
        ax.plot(*zip(*negative_precision_recall_points))
        plt.title('Precision versus Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()


    # Evaluate performance on test data 
    def Eval(self, test):
        print("Predicting test reviews labels ...")
        Y_pred = self.PredictLabel(test.X)
        print("Evaluate the NaiveBayes model ...")
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    def FeaturesSelection(self, vocab):
        word_neg = defaultdict(float)
        word_pos = defaultdict(float)
        for word in self.logP_positive:
            #Compute Words weight regarding a specific label
            word_neg[vocab.GetWord(word)] = (self.logP_negative[word] - self.logP_positive[word]) * self.count_negative[word] - self.count_positive[word]
            word_pos[vocab.GetWord(word)] = (self.logP_positive[word] - self.logP_negative[word]) * self.count_positive[word] - self.count_negative[word]
        word_neg = sorted(word_neg.items(), key=lambda x: x[1], reverse=True)
        word_pos = sorted(word_pos.items(), key=lambda x: x[1], reverse=True)
        #Print words ranks
        print("Top 20 Negative Words: \n", word_neg[:20])
        print("Top 20 Positive Words: \n", word_pos[:20])


if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    indexes = range(0,25000)
    positive_probs, negative_probs = nb.PredictProb(testdata, indexes)
    print("Precision/Recall curve creation ...")
    #This one will take time to process regarding the fact that we compute the entire dataset for both label.#
    nb.Precision_Recall_curve(testdata, positive_probs, negative_probs, indexes)
    nb.FeaturesSelection(traindata.vocab)

