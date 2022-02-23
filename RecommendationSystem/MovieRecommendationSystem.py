import os

path = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import gdown
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from annoy import AnnoyIndex
import zipfile
import time
from google.colab import drive

# import os
#
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))


### consider content model based

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm_notebook
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

data = pd.read_csv('./RecommendationSystem/netflix_list.csv')

data.head()

data.info()
data.describe()

data = data[['imdb_id', 'title', 'rating']]

data = data.dropna()

data_pivot = data.pivot_table(columns='imdb_id', index='title', values='rating')

data_pivot.fillna(0, inplace=True)

from scipy.sparse import csr_matrix

data_sparce = csr_matrix(data_pivot)

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(algorithm='brute')

model.fit(data_sparce)

distance, suggestion = model.kneighbors(data_pivot.iloc[100, :].values.reshape(1, -1))

for i in range(len(suggestion)):
    print(data_pivot.index[suggestion[i]])

from sklearn.metrics import confusion_matrix
