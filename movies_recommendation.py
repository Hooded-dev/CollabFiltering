#! /usr/bin/python3

from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from surprise import Reader, Dataset, SVD
import pickle
import re
import sys

file = sys.argv[1]

rat = pd.read_csv(file)
r = rat.shape
reader = Reader()
data = Dataset.load_from_df(rat[['userId', 'movieId', 'rating']], reader)


svd = pickle.load(open('movie_recommendation.pkl', 'rb'))

data = pd.DataFrame([])
for i in range(r[0]):
    a = rat["userId"][i]
    b = rat["movieId"][i]
    c = rat["rating"][i]
    y = svd.predict(a,b,c)
    t = y[:4]
    df = pd.DataFrame([t])
    df = df.drop(columns=df.columns[2])
    df = df.drop(columns=df.columns[0])
    xx = df.to_numpy()
    data = data.append(pd.DataFrame(xx), ignore_index=True)

res = data.groupby(data.iloc[:,0]).mean()
print(res)

