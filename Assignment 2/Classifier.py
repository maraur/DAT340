import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate



with open("a2_train_final.tsv", encoding='utf8') as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    rs = list(rd)
    #for row in rs:
    #    print(row)


#tokenize_re = re.compile(r'''
#                         \d+[:\.]\d+
#                         |(https?://)?(\w+\.)(\w{2,})+([\w/]+)
#                         |[@\#]?\w+(?:[-']\w+)*
#                         |[^a-zA-Z0-9 ]+''',
#                         re.VERBOSE)

def tokenize(text):
    return [ m.group() for m in tokenize_re.finditer(text) ]

def read_feature_descriptions(filename):
    names = []
    types = []
    with open(filename, encoding='utf8') as f:
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

feat_names, feat_types = read_feature_descriptions('a2_train_final.tsv')

def read_data(filename, feat_names, feat_types):
    X = []
    Y = []
    with open(filename, encoding='utf8') as fl:
        f = csv.reader(fl, delimiter="\t", quotechar='"')
        for l in f:
            cols = l.strip('\n.').split(', ')
            if len(cols) < len(feat_names): # skip empty lines and comments
                continue
            X.append( { n:t(c) for n, t, c in zip(feat_names, feat_types, cols) } )
            Y.append(cols[-1])
    return X, Y

Xtrain, Ytrain = read_data('a2_train_final.tsv', feat_names, feat_types)
Xtest, Ytest = read_data('a2_train_final.tsv', feat_names, feat_types)

pipeline = make_pipeline(
    DictVectorizer(),
    DummyClassifier()
)
print(train)
#print(cross_validate(pipeline, Xtrain, Ytrain))