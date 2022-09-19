# %% [markdown]
# ## K-nearest neighbour classification algorithm model

# %%
# Importing all necessary packages
import matplotlib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd 
import joblib as extractor

# %% [markdown]
# Importing the data from the CSV dataset using Pandas

# %%
dataframe = pd.read_csv("data/person-age-gender-games-dataset.csv")

# %% [markdown]
# Standarising the dataset by changing the male and female as 0 and 1 for KNN classification algo

# %%
dataframe['Gender'] = dataframe['Gender'].replace({"female": 0, "male": 1})
dataframe.head()

# %% [markdown]
# Spliting the features and label to x and y variable

# %%
x = dataframe.drop(columns=['Person','Favorite Game'])
y = dataframe['Favorite Game']

# %% [markdown]
# Spliting the train data and the test data from the given dataset

# %%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

# %% [markdown]
# Initialising the KNN classifier and fitting the training data to train the model

# %%
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# %% [markdown]
#  Predicting the model by the test data

# %%
predict = knn.predict(x_test)
print(predict)
print("Accuracy : " + str(accuracy_score(y_test, predict)))
print("Confusion Matrix : \n" + str(confusion_matrix(y_test, predict)))

extractor.dump(knn,"KNN Person sport interest - model.joblib")


