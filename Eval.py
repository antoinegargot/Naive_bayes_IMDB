import numpy as np

class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold
        #create a confusion matrix
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def Accuracy(self):
        return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))

    def Precision(self):
    	#Compute precision of the confusion matrix
        if ((self.tp + self.fp) != 0):
            return self.tp / (self.tp + self.fp)
        else:
            return 1

    def Recall(self):
    	#Compute recall of the confusion matrix
        if ((self.tp + self.fn) != 0):
            return self.tp / (self.tp + self.fn)
        else:
            return 1

    def ComputeConfusionMatrix(self):
        for pred, label in zip(self.pred, self.gold):
            if pred + label == 2:
                self.tp += 1
            elif(pred + label) == -2:
                self.tn += 1
            elif pred > label:
                self.fp += 1
            else:
                self.fn += 1