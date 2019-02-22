import pandas as pd
import numpy as np
df = pd.read_csv("a2_train_final.tsv",sep="\t")
#print(data.values)
vals = df.values

trueAnn = 0
falseAnn = 0
currRow = 0
print("Size of data set is", df.values.size/2)

for row in vals:
    v = row[0].split('/')
    if v.count('1') > (len(v)/2) or v.count('0') > (len(v)/2):
        trueAnn += 1
    else:
        falseAnn += 1
        df = df.drop(df.index[1])
    currRow += 1

print("Consistent annotations: ", trueAnn, ", Inconsistent annotations: ", falseAnn)
print("Size of data set is", df.values.size/2)