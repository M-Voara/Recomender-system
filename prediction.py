from sparse_matrix import*
import csv
import numpy as np
from defintion import*
from math import sqrt

path_movies = 'path to movies.csv'

dict_movies= {}
with open(path_movies, 'r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',')
    next(data_reader, None)
    for row in data_reader:
        movie_sys, title, genre = row
        dict_movies[movie_sys] = title


movie_name = input("Enter your movie:")
number_of_user = len(user_map_list)
number_of_movie = len(movie_map_list)
latent_dimension = 5
lamda = 0.001
tau = 0.01
gama = 0.5
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
    
  
movie_to_name = []
for i in movie_map_list:
    movie_to_name.append(dict_movies[i])

def getname(movie_index, movie_map_list, dict_movies):
    # Get the movie ID in the 'movie_map_list' based on the provided 'movie_index'
    movie_sys = movie_map_list[movie_index]
    
    # Use the 'movie_sys' ID to look up the movie name in the 'dict_movies_25m' dictionary
    return dict_movies[movie_sys]


def searchmovie(search):
    List_of_movie = []  # Initialize a list to store matching movie indices and names
    
    # Iterate through the list of movie names and their indices
    for i, name in enumerate(movie_to_name):
        if search in name:  # Check if the search keyword is a substring of the movie name
            # If there's a match, add the movie's index and name to the list
            List_of_movie.append((i, name))
    
    return List_of_movie  # Return the list of matching movie indices and names


def predict(movie_index,movie_vector,movie_bias):
    """
    Returns a list of tuple (movie_index,score) of top 10 recommendations for the choosen movie

    """

    #Create a dummy user
    dummy_user_vector= movie_vector[movie_index].copy()
    
    #Get the prediction of all movie vectors
    recomender = [(i,(dummy_user_vector @ movie_vector[i]) + 0.05*movie_bias[i]) for i in range(len(movie_vector))]

    #Sort the recomender from the worse to the best
    recomender = sorted(recomender, key=lambda x: x[1])

    #Return the top 10 recommendations
    recomender = recomender[-5:]
    
    return recomender

if not searchmovie(movie_name):
    print("No suggestion")
else:
    for i in predict(searchmovie(movie_name)[0][0],movie_vector,movie_bias):
        print(i[0],getname(i[0],movie_map_list,dict_movies))

