
# coding: utf-8

# In[24]:

import pandas as pd

board_games = pd.read_csv("G:/DataQuest/games.csv")
#print(board_games.head(5))
print(board_games.shape)
board_games=board_games.dropna(axis=0)
print(board_games.shape)
board_games = board_games[board_games["users_rated"] != 0]
print(board_games.shape)
#print(board_games.columns)


# In[13]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(board_games["average_rating"])


# In[17]:

std_dev = board_games["average_rating"].std()
print("Standard deviation:", std_dev)
mean = sum(board_games["average_rating"])/len(board_games["average_rating"])
print("Mean:", mean)


# In[29]:

from sklearn.cluster import KMeans

kmean_model = KMeans(n_clusters=5)

cols = list(board_games.columns)
cols.remove("name")
cols.remove("id")
cols.remove("type")

numeric_columns = board_games[cols]
#print(numeric_columns.columns)

kmean_model.fit(numeric_columns)
labels = kmean_model.labels_
print(labels)


# In[35]:

import numpy as np

game_mean = numeric_columns.apply(np.mean,axis=1)
#print(game_mean.head())
game_std = numeric_columns.apply(np.std, axis=1)
#print(game_std.head())

plt.scatter(game_mean, game_std, c=labels)
plt.show()


# In[36]:

correlations = numeric_columns.corr()
print(correlations["average_rating"])


# In[43]:

coll = list(numeric_columns.columns)
#coll.remove("bayes_average_rating")
#coll.remove("minplayers")
#coll.remove("maxplayers")

numeric_columns = numeric_columns[coll]
#print(numeric_columns.head())


# In[47]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression()

pred_col = list(numeric_columns.columns)
pred_col.remove("average_rating")

reg.fit(board_games[pred_col], board_games["average_rating"])

predictions = reg.predict(board_games[pred_col])
#print(predictions)
error = mean_squared_error(board_games["average_rating"], predictions)
print(error)


# In[ ]:



