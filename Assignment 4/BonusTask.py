
"""This file shows a couple of implementations of the perceptron learning
algorithm. It is based on the code from Lecture 3, but using the slightly
more compact perceptron formulation that we saw in Lecture 6.

There are two versions: Perceptron, which uses normal NumPy vectors and
matrices, and SparsePerceptron, which uses sparse vectors and matrices.
The latter may be faster when we have high-dimensional feature representations
with a lot of zeros, such as when we are using a "bag of words" representation
of documents.
"""

import numpy as np
from sklearn.base import BaseEstimator
import random
import scipy as sp

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)
        A = []
        for row in scores:
            A.append(np.argmax(row))

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        out = self.decode_multi_outputs(A)
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])

    def find_number_of_classes(self, Y):
        i = 0
        classes = sorted(set(Y))
        for c in classes:
            print(c, "has number", i)
            i += 1
        return len(classes)

    def encode_multi_outputs(self, Y):
        switcher = {
            "books": 0,
            "camera": 1,
            "dvd": 2,
            "health": 3,
            "music": 4,
            "software": 5
        }
        Ye = list(map(switcher.get, Y))
        return Ye

    def decode_multi_outputs(self, Y):
        switcher = {
            0: "books",
            1: "camera",
            2: "dvd",
            3: "health",
            4: "music",
            5: "software"
        }
        Yd = list(map(switcher.get, Y))
        return Yd
##### The following part is for the optional task.

### Sparse and dense vectors don't collaborate very well in NumPy/SciPy.
### Here are two utility functions that help us carry out some vector
### operations that we'll need.

def add_sparse_to_dense(x, w, factor):
    """
    Adds a sparse vector x, scaled by some factor, to a dense vector.
    This can be seen as the equivalent of w += factor * x when x is a dense
    vector.
    """
    w[x.indices] += factor * x.data

def sparse_dense_dot(x, w):
    """
    Computes the dot product between a sparse vector x and a dense vector w.
    """
    return np.dot(w[x.indices], x.data)


"""
-----------------------------------------------------------------------------------------------------------------------
----------------------------------------- Our code follows from here --------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
"""

class MultiClassSVM(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=100000):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y, lmbd=0.001):
        """
        Train a linear classifier using the pegasos learning algorithm.
        """
        self.RegPar = lmbd

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        #self.find_classes(Y)
        classNumber = self.find_number_of_classes(Y)
        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        #Ye = self.encode_outputs(Y)
        Ye = self.encode_multi_outputs(Y)
        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros((n_features, classNumber))

        L = list(zip(X, Ye))

        for i in range(self.n_iter):
            x, y = random.choice(L)
            t = i + 1
            lr = 1/(t*self.RegPar)
            # Compute the output score for this instance.
            scoreYi = x.dot(self.w[:, y])
            A = []
            for j in range(classNumber):
                scoreY = x.dot(self.w[:, j])
                if y == j:
                    A.append(0 - scoreYi + scoreY)
                else:
                    A.append(1 - scoreYi + scoreY)
            yHat = np.argmax(A)
            m1 = np.zeros((n_features, classNumber))
            m2 = np.zeros((n_features, classNumber))
            m1[:, yHat] = x
            m2[:, y] = x
            self.w = (1 - lr * self.RegPar) * self.w - lr * (m1 - m2)



class MultiClassLR(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=100000):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y, lmbd=0.01):
        """
        Train a linear classifier using the pegasos learning algorithm.
        """
        self.RegPar = lmbd

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        #self.find_classes(Y)
        classNumber = self.find_number_of_classes(Y)
        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        #Ye = self.encode_outputs(Y)
        Ye = self.encode_multi_outputs(Y)
        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros((n_features, classNumber))

        L = list(zip(X, Ye))

        for i in range(self.n_iter):
            x, y = random.choice(L)
            t = i + 1
            lr = 1/(t*self.RegPar)
            # Compute the output score for this instance.
            scores = x.dot(self.w)
            subGrad = np.zeros((n_features, classNumber))
            p = sp.special.softmax(scores)
            phi2 = np.zeros((n_features, classNumber))
            phi2[:, y] = x
            for r in range(classNumber):
                phi1 = np.zeros((n_features, classNumber))
                phi1[:, r] = x
                subGrad += p * phi1 - phi2
                
            self.w = (1 - lr * self.RegPar) * self.w - lr * subGrad
