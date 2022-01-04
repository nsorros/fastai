from fastai.tabular.all import *
import pandas as pd

path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(
    path/"adult.csv",
    path=path,
    cat_names=["workclass", "education", "marital-status", "occupation", "relationship", "race"],
    cont_names=["age", "fnlwgt", "education-num"],
    y_names="salary",
    procs=[Categorify, FillMissing, Normalize]
)
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(3)
