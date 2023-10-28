import numpy as np
import csv
import matplotlib.pyplot as plt
from math import sqrt

def write_data(data,data_name):
    with open(data_name, "wb") as file:
        pickle.dump(data, file)
    
    
def load_data(data_name):
    with open(data_name, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def calculate_loss_and_rmse(user_vector, 
                            movie_vector, 
                            sparse_user, 
                            sparse_movie, 
                            user_bias, 
                            movie_bias, 
                            lamda, 
                            tau, 
                            gama):
  loss = 0
  num_rating = 0
  
  # Calculate the loss by iterating through users and their rated movies
  for m in range(len(sparse_user)):
    for movie_id, rating in sparse_user[m]:
      # Compute the squared error for each rating prediction
      loss += (rating - (user_vector[m].T @ movie_vector[movie_id] + user_bias[m] + movie_bias[movie_id]))**2
      num_rating += 1

  reg_user = 0
  
  # Calculate regularization term for user vectors
  for m in range(len(sparse_user)):
    reg_user += user_vector[m].T @ user_vector[m]

  reg_movie = 0
  
  # Calculate regularization term for movie vectors
  for n in range(len(sparse_movie)):
    reg_movie += movie_vector[n].T @ movie_vector[n]

  # Calculate the final loss with regularization terms and bias terms
  loss_final = (-lamda/2) * loss - (tau/2) * reg_user - (tau/2) * reg_movie - (gama/2) * (np.dot(user_bias, user_bias)) - (gama/2) * (np.dot(movie_bias, movie_bias))
  
  # Calculate the mean squared error (RMSE) using the computed loss and the number of ratings
  mse = loss / num_rating
  
  # Return the negative loss (as this is often minimized), and the square root of the mean squared error (RMSE)
  return -loss_final, np.sqrt(mse)


def plot_power_law(p, k):
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()

    # Scatter plot for the values in 'p' and their counts
    ax.scatter(p, [p.count(i) for i in p], label='p')

    # Scatter plot for the values in 'k' and their counts
    ax.scatter(k, [k.count(i) for i in k], label='k')

    # Set the y-axis to a logarithmic scale
    plt.yscale('log')

    # Set the x-axis to a logarithmic scale
    plt.xscale('log')

    # Plot the data
    plt.plot()

    # Show the plot
    plt.show()


def update_user(m, 
                latent_dimension, 
                sparse_user_train, 
                movie_vector, user_bias, 
                movie_bias, 
                lamda, 
                tau):
  matrix_left = 0  # Initialize the left part of the matrix equation
  matrix_right = 0  # Initialize the right part of the matrix equation
  
  # Iterate through movies rated by user 'm' in the training data
  for movie_id, rating in sparse_user_train[m]:
    # Calculate the left part of the matrix equation by taking the outer product of movie vectors
    matrix_left += np.outer(movie_vector[movie_id], movie_vector[movie_id])
    
    # Calculate the right part of the matrix equation, including rating, user bias, and movie bias
    matrix_right += movie_vector[movie_id] * (rating - user_bias[m] - movie_bias[movie_id])

  # Regularize the right part of the matrix equation with 'lambda'
  matrix_right = lamda * matrix_right
  
  # Regularize the left part of the matrix equation with 'lambda' and add a regularization term 'tau'
  matrix_left = (lamda * matrix_left + tau * np.identity(latent_dimension))
  
  # Compute the inverse of the 'matrix_left'
  inv = np.linalg.inv(matrix_left)
  
  # Solve for the updated user vector and return it
  return inv @ matrix_right


def update_movie(n, 
                latent_dimension, 
                sparse_movie_train, 
                user_vector, 
                user_bias, 
                movie_bias, 
                lamda, 
                tau):
  matrix_left = 0  # Initialize the left part of the matrix equation
  matrix_right = 0  # Initialize the right part of the matrix equation
  
  # Iterate through users who have rated movie 'n' in the training data
  for user_id, rating in sparse_movie_train[n]:
    # Calculate the left part of the matrix equation by taking the outer product of user vectors
    matrix_left += np.outer(user_vector[user_id], user_vector[user_id])
    
    # Calculate the right part of the matrix equation, including rating, movie bias, and user bias
    matrix_right += user_vector[user_id] * (rating - movie_bias[n] - user_bias[user_id])

  # Regularize the right part of the matrix equation with 'lambda'
  matrix_right = lamda * matrix_right
  
  # Regularize the left part of the matrix equation with 'lambda' and add a regularization term 'tau'
  matrix_left = (lamda * matrix_left + tau * np.identity(latent_dimension))
  
  # Compute the inverse of the 'matrix_left'
  inv = np.linalg.inv(matrix_left)
  
  # Solve for the updated movie vector and return it
  return inv @ matrix_right




def update_user_bias(m, 
                    sparse_user_train, 
                    user_vector, 
                    movie_vector, 
                    movie_bias, 
                    lamda, 
                    gama):
    bias = 0  # Initialize the user bias value
    
    # Iterate through movies rated by user 'm' in the training data
    for movie_id, rating in sparse_user_train[m]:
        # Calculate the bias for the user by comparing the actual rating with the predicted rating
        bias += rating - (user_vector[m].T @ movie_vector[movie_id] + movie_bias[movie_id])

    # Compute and return the updated user bias, considering regularization terms 'lambda' and 'gamma'
    return (lamda * bias) / (lamda * len(sparse_user_train[m]) + gama)



def update_movie_bias(n, 
                    sparse_movie_train, 
                    movie_vector, 
                    user_vector, 
                    user_bias, 
                    lamda, 
                    gama):
    bias = 0  # Initialize the movie bias value
    
    # Iterate through users who have rated movie 'n' in the training data
    for user_id, rating in sparse_movie_train[n]:
        # Calculate the bias for the movie by comparing the actual rating with the predicted rating
        bias += rating - (movie_vector[n].T @ user_vector[user_id] + user_bias[user_id])

    # Compute and return the updated movie bias, considering regularization terms 'lambda' and 'gamma'
    return (lamda * bias) / (lamda * len(sparse_movie_train[n]) + gama)


def calculate_loss_and_rmse(user_vector, 
                            movie_vector, 
                            sparse_user, 
                            sparse_movie, 
                            user_bias, 
                            movie_bias, 
                            lamda, 
                            tau, 
                            gama):
    loss = 0  # Initialize the loss
    num_rating = 0  # Initialize the count of ratings
    
    # Calculate the loss by iterating through users and their rated movies
    for m in range(len(sparse_user)):
        for movie_id, rating in sparse_user[m]:
            # Compute the squared error for each rating prediction
            loss += (rating - (user_vector[m].T @ movie_vector[movie_id] + user_bias[m] + movie_bias[movie_id]))**2
            num_rating += 1

    reg_user = 0  # Initialize the regularization term for user vectors
    
    # Calculate regularization term for user vectors
    for m in range(len(sparse_user)):
        reg_user += user_vector[m].T @ user_vector[m]

    reg_movie = 0  # Initialize the regularization term for movie vectors
    
    # Calculate regularization term for movie vectors
    for n in range(len(sparse_movie)):
        reg_movie += movie_vector[n].T @ movie_vector[n]

    # Calculate the final loss with regularization terms and bias terms
    loss_final = (-lamda/2)*loss - (tau/2)*reg_user - (tau/2)*reg_movie - (gama/2)*(np.dot(user_bias, user_bias)) - (gama/2)*(np.dot(movie_bias, movie_bias))
    
    mse = loss / num_rating  # Calculate the mean squared error (MSE)
    
    # Return the negative loss (as this is often minimized) and the square root of the mean squared error (RMSE)
    return -loss_final, np.sqrt(mse)



def plot_rmse(list_of_iteration, 
            rmse_history_train, 
            rmse_history_test):
    plt.plot(list_of_iteration,rmse_history_train, label = 'Data train', marker = '.')
    plt.plot(list_of_iteration,rmse_history_test, label = 'Data test', marker = '.')
    plt.legend()
    plt.title("RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.plot()
    plt.show()



def plot_loss(list_of_iteration, 
            losses_history_train, 
            losses_history_test):
    fig, ax = plt.subplots()
    plt.plot(list_of_iteration,losses_history_train, label = 'Data train', marker = '.')
    plt.plot(list_of_iteration,losses_history_test, label = 'Data test', marker = '.')
    plt.legend()
    plt.title("Negative likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot()
    plt.show()



