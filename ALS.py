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
list_of_iteration =[]
rmse_history_train =[]
rmse_history_test = []
losses_history_train =[]
losses_history_test = []


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
    

  LOSS_TRAIN , RMSE_TRAIN = calculate_loss_and_rmse(user_vector, movie_vector, sparse_user_train, sparse_movie_train, user_bias, movie_bias, lamda, tau, gama)
  LOSS_TEST , RMSE_TEST = calculate_loss_and_rmse(user_vector, movie_vector, sparse_user_test, sparse_movie_test, user_bias, movie_bias, lamda, tau, gama)
  rmse_history_test.append( RMSE_TEST)
  rmse_history_train.append(RMSE_TRAIN)
  losses_history_train.append(LOSS_TRAIN)
  losses_history_test.append(LOSS_TEST)
  list_of_iteration.append(iteration)
  print(f'Iteration{iteration +1 }, rmse_train ={RMSE_TRAIN}, rmse_test = {RMSE_TEST}, Loss_Train ={LOSS_TRAIN}, Loss_Test = {LOSS_TEST}')

plot_rmse(list_of_iteration,rmse_history_train,rmse_history_test)
plot_loss(list_of_iteration, losses_history_train, losses_history_test)