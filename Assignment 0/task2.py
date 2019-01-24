from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def read_feature_descriptions(filename):
    names = []
    types = []
    with open(filename) as f:
        for l in f:
            if l[0] == '|' or ':' not in l:
                continue
            cols = l.split(':')
            names.append(cols[0])
            if cols[1].startswith(' continuous.'):
                types.append(float)
            else:
                types.append(str)
    return names, types

feat_names, feat_types = read_feature_descriptions('datasets/adult.names')

def read_data(filename, feat_names, feat_types):
    X = []
    Y = []
    with open(filename) as f:
        for l in f:
            cols = l.strip('\n.').split(', ')
            if len(cols) < len(feat_names): # skip empty lines and comments
                continue
            X.append( { n:t(c) for n, t, c in zip(feat_names, feat_types, cols) } )
            Y.append(cols[-1])
    return X, Y

Xtrain, Ytrain = read_data('datasets/adult.data', feat_names, feat_types)
Xtest, Ytest = read_data('datasets/adult.test', feat_names, feat_types)


pipeline = make_pipeline(
    DictVectorizer(),
    GradientBoostingClassifier()
)

print(cross_validate(pipeline, Xtrain, Ytrain))

pipeline.fit(Xtrain, Ytrain)
Yguess = pipeline.predict(Xtest)
print(accuracy_score(Ytest, Yguess))