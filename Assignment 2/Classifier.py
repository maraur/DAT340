import csv
import numpy as np
import pandas as pd


with open("a2_train_final.tsv", encoding='utf8') as fd:
    df = pd.read_csv(fd, sep="\t")
    rf = df.values
    count = 0
    for row in rf:
        A = row[0].split('/')
        if A.count('1') > len(A)/2 or A.count('0') > len(A)/2:
            count += 1
        else:
            df.drop([row[0], row[1]])
