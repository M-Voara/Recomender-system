import csv
import matplotlib.pyplot as plt


path_ratings = 'path to ratings.csv'
ratings = []
with open(path_ratings, 'r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',')
    next(data_reader, None)
    for row in data_reader:
        user_id, movie_id, rating, timestamp = row
        ratings.append([user_id, movie_id, float(rating)])


def sparse_matrix(data):
  user_map_dict = {}  # A dictionary to map user IDs to unique indices
  user_map_list = []  # A list to store user IDs in order

  movie_map_dict = {}  # A dictionary to map movie IDs to unique indices
  movie_map_list = []  # A list to store movie IDs in order

  sparse_user = []  # A list to represent the sparse matrix for users
  sparse_movie = []  # A list to represent the sparse matrix for movies

  # Iterate through the input data to create user and movie mappings and populate sparse matrices
  for user_id, movie_id, rating in data:
    if user_id not in user_map_dict:
      # Add the user to the list and assign a unique index
      user_map_list.append(user_id)
      user_map_dict[user_id] = len(user_map_dict)
      sparse_user.append([])

    if movie_id not in movie_map_dict:
      # Add the movie to the list and assign a unique index
      movie_map_list.append(movie_id)
      movie_map_dict[movie_id] = len(movie_map_dict)
      sparse_movie.append([])

    # Add the rating to the corresponding user-movie pair in the sparse matrices
    sparse_user[user_map_dict[user_id]].append((movie_map_dict[movie_id], rating))
    sparse_movie[movie_map_dict[movie_id]].append((user_map_dict[user_id], rating))

  # Return user and movie mappings along with the sparse matrices
  return user_map_dict, user_map_list, movie_map_dict, movie_map_list, sparse_user, sparse_movie



user_map_dict,user_map_list,movie_map_dict,movie_map_list,sparse_user,sparse_movie = sparse_matrix(ratings)

movie_rated_per_user = [len(a) for a in sparse_user]
users_rating_movie = [len(a) for a in sparse_movie]


fig, ax = plt.subplots(figsize = (10,6))

ax.scatter(movie_rated_per_user,[movie_rated_per_user.count(i) for i in movie_rated_per_user],marker = '+', label = 'Movie')
ax.scatter(users_rating_movie,[users_rating_movie.count(i) for i in users_rating_movie], marker = 'v', label = 'user')

plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree')
plt.ylabel('Frequencies')
plt.title('Power law')

plt.plot()
plt.show()