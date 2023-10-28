import csv
path_ratings = 'path to ratings.csv'

ratings = []
with open(path_ratings, 'r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',')
    next(data_reader, None)
    for row in data_reader:
        user_id, movie_id, rating, timestamp = row
        ratings.append([user_id, movie_id, float(rating)])



#Sparse Matrix
def sparse_matrix(data):
  #Initialize dictionaries and lists to map user and movie IDs to unique indices
  user_map_dict = {}
  user_map_list = []

  movie_map_dict = {}
  movie_map_list = []
  #Iterate through the data to create user and movie mappings
  for user_id, movie_id, rating in data :
    if user_id not in user_map_dict:
      #Add the user to the list and assign a unique index
      user_map_list.append(user_id)
      user_map_dict[user_id]= len(user_map_dict)
    
    if movie_id not in movie_map_dict:
      # Add the movie to the list and assign a unique index
      movie_map_list.append(movie_id)
      movie_map_dict[movie_id] = len(movie_map_dict)
      
  #Initialize sparse matrices for training and testing data
  sparse_user_test = [[] for i in range(len(user_map_list))]
  sparse_user_train = [[] for i in range(len(user_map_list))]

  sparse_movie_test = [[] for i in range(len(movie_map_list))]
  sparse_movie_train = [[] for i in range(len(movie_map_list))]

  #Fill in the sparse matrices with user-movie ratings
  for index in range(len(data)):
    user_num = data[index][0]
    movie_num = data[index][1]
    rating = data[index][2]

    user_index = user_map_dict[user_num]
    movie_index = movie_map_dict[movie_num]
    if index < 0.9*len(data):
      #For the training set, add ratings to user-movie pairs
      sparse_user_train[user_index].append((movie_index, float(rating)))
      sparse_movie_train[movie_index].append((user_index, float(rating)))
    else:
      #For the test set, add ratings to user-movie pairs
      sparse_user_test[user_index].append((movie_index, float(rating)))
      sparse_movie_test[movie_index].append((user_index, float(rating)))
  #Return the mappings and sparse matrices
  return user_map_dict,user_map_list,movie_map_dict,movie_map_list,sparse_user_train, sparse_user_test, sparse_movie_train, sparse_movie_test


user_map_dict,user_map_list,movie_map_dict,movie_map_list, sparse_user_train, sparse_user_test, sparse_movie_train, sparse_movie_test = sparse_matrix(ratings)