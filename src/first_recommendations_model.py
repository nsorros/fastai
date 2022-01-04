from fastai.data.external import untar_data, URLs
from fastai.collab import *
import pandas as pd

path = untar_data(URLs.ML_SAMPLE)
data = pd.read_csv(path/"ratings.csv")
print(data.head())

dls = CollabDataLoaders.from_df(data)
learn = collab_learner(dls, y_range=(0.5,5.5))
print(learn.fine_tune(10))
