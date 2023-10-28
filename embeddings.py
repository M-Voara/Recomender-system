import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from defintion import*
from sparse_matrix import*

number_of_user = len(user_map_list)
number_of_movie = len(movie_map_list)
latent_dimension = 16
lamda = 0.01
tau = 0.01
gama = 0.05
sigma = 1/sqrt(latent_dimension)
user_bias = np.zeros(number_of_user)
movie_bias = np.zeros(number_of_movie)
user_vector = np.random.normal(0,sigma,size=[number_of_user,latent_dimension])
movie_vector = np.random.normal(0,sigma,size=[number_of_movie,latent_dimension])

nbr_iteration = 10
for iteration in range(nbr_iteration):
  # update user_bias
  for m in range(number_of_user):
    user_bias[m] = update_user_bias(m, sparse_user_train, user_vector, movie_vector, movie_bias, lamda, gama)
  # update user_vector
  for m in range(number_of_user):
    if not sparse_user_train[m]:
      continue
    user_vector[m] = update_user(m, latent_dimension, sparse_user_train, movie_vector, user_bias, movie_bias, lamda, tau)
  # update movie_bias
  for n in range(number_of_movie):
     movie_bias[n] = update_movie_bias(n, sparse_movie_train, movie_vector, user_vector, user_bias, lamda, gama)
  # update movie_vector
  for n in range(number_of_movie):
    if not sparse_movie_train[n]:
      continue
    movie_vector[n] = update_movie(n, latent_dimension, sparse_movie_train, user_vector, user_bias, movie_bias, lamda, tau)

def coordinates(genre_movie):
    x = []
    y = []
    for i in genre_movie:
        x.append(movie_vector[i][0])
        y.append(movie_vector[i][1])
    return x,y


movie1 = [2148, 7613, 2678, 3082, 2947]
x_movie1, y_movie1 = coordinates(movie1)


movie2 = [2497, 4430, 166, 761, 1290]
x_movie2, y_movie2 = coordinates(movie2)

movie3 = [2218,807,3656,18,655]
x_movie3, y_movie3 = coordinates(movie3)

movie4 = [3966,5343,4969,4286,6009]
x_movie4, y_movie4 = coordinates(movie4)


def plot_enmbedings(ax, x, y, genre_movie, genre, label_offset, marker):
    ax.scatter(x, y, label = str(genre), marker = marker)
    ax.legend()

fig, ax = plt.subplots(figsize=(8, 8))

# Define your data and categories
categories = [
    (x_movie1, y_movie1, movie1, 'movie1'),
    (x_movie2, y_movie2, movie2, 'movie2'),
    (x_movie3, y_movie3, movie3,'movie3'),
    (x_movie4, y_movie4, movie4,'movie4'),
    
]

plt.title("Embeddings")
label_offset = -0.04  # Adjust this value as needed
markers = ['s', 'x', 'h', '^']
for i,category_data in enumerate(categories):
    x, y, movie_ids, genre = category_data
    plot_enmbedings(ax, x, y, movie_ids, genre, label_offset,markers[i])

ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Définissez les positions des ticks
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot()
plt.show()