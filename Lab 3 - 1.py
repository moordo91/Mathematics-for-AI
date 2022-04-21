# import essential packages
import numpy as np
import pandas as pd

# load Movielens dataset
train_datapath = "data/movielens_latest_small_train.csv"
test_datapath = "data/movielens_latest_small_test.csv"
rating_train = pd.read_csv(train_datapath)
rating_test = pd.read_csv(test_datapath)


# --------------------------------------------------------------------------
# Useful functions
# --------------------------------------------------------------------------
def evaluate_rmse(rating_true, rating_pred):
    """ Compute the root mean squared error between labels and predictions
    Args:
        rating_true: rating labels for the test set (pandas dataframe)
        rating_pred: rating matrix (numpy array) 
    Returns:
        rmse: (float)
    """
    mse = 0.0
    rating_pred = np.clip(rating_pred, 0.5, 5.0)
    for i in range(len(rating_true)):
        u = rating_true["user_id"][i]
        m = rating_true["movie_id"][i]
        r_true = rating_true["rating"][i]
        r_pred = rating_pred[u, m]
        mse += (r_true - r_pred)**2
    rmse = np.sqrt(mse / len(rating_true))
    return rmse

def compute_means(rating, num_users, num_movies):
    """ Compute mean, user_mean, and movie_mean from training dataset
    Args:
        rating: rating labels for the training set (pandas dataframe)
        num_users: (int)
        num_movies: (int) 
    Returns:
        mean: the average rating of the training set (float)
        user_mean: the average rating for each user (numpy array)
        movie_mean: the average rating for each movie (numpy array)
    """
    mean = rating["rating"].mean()
    
    user_mean = np.zeros((num_users, 1))
    user_count = np.zeros((num_users, 1))
    movie_mean = np.zeros((num_movies, 1))
    movie_count = np.zeros((num_movies, 1))
    for i in range(len(rating)):
        u = rating["user_id"][i]
        m = rating["movie_id"][i]
        r = rating["rating"][i]
        user_mean[u] += r
        user_count[u] += 1
        movie_mean[m] += r
        movie_count[m] += 1
    
    for i in range(num_movies):
        if movie_count[i] < 1:
            movie_mean[i] = mean
            movie_count[i] = 1

    user_mean = (user_mean / user_count) - mean
    movie_mean = (movie_mean / movie_count) - mean
    
    return mean, user_mean, movie_mean

def fill_rating(matrix, rating):
    """ Fill labeled ratining to rating matrix
    Args:
        matrix: ratinig matrix (numpy array)
        rating: rating labels for the training set (pandas dataframe)
    Returns:
        matrix: ratinig matrix (numpy array)
    """
    for i in range(len(rating)):
        u = rating["user_id"][i]
        m = rating["movie_id"][i]
        matrix[u, m] = rating["rating"][i]
    return matrix

def add_mean_values(matrix_norm, mean, user_mean, movie_mean):
    """ Add mean values to normalized rating 
    Args:
        matrix_norm: normalized rating matrix (numpy array)
        num_users: (int)
        num_movies: (int) 
    Returns:
        matrix: rating matrix (numpy array)
    """
    matrix = matrix_norm.copy()
    matrix += mean
    for i in range(len(user_mean)):
        matrix[i, :] += user_mean[i]
    for i in range(len(movie_mean)):
        matrix[:, i] += movie_mean[i]
    return matrix

# --------------------------------------------------------------------------
# Useful variables
# --------------------------------------------------------------------------
num_users = 610
num_movies = 9742

mean, user_mean, movie_mean = compute_means(rating_train, num_users, num_movies)

rating_train_norm = rating_train.copy()
for i in range(len(rating_train)):
    u = rating_train["user_id"][i]
    m = rating_train["movie_id"][i]
    rating_train_norm["rating"][i] -= (mean + user_mean[u] + movie_mean[m])

# rating_train_norm contains normalized rating for problem 2 ~ 3 


# -------------------------------------------------------------------- 
# Problem 1: Recommendation with Mean Values
# --------------------------------------------------------------------

# Prediction with mean 
rating_mean = mean * np.ones((num_users, num_movies))
rmse_mean_test = evaluate_rmse(rating_test, rating_mean)
print(f"RMSE - mean: {rmse_mean_test:.6f}")

# --------------------------------------------------------------------
# Problem 1-1: Prediction with mean and user_mean 
# --------------------------------------------------------------------
rating_user_mean = mean * np.ones((num_users, num_movies))

# Fill this

rmse_user_mean_test = evaluate_rmse(rating_test, rating_user_mean)
print(f"RMSE - user_mean: {rmse_user_mean_test:.6f}")

# --------------------------------------------------------------------
# Problem 1-2: Prediction with mean and movie_mean
# --------------------------------------------------------------------
rating_movie_mean = mean * np.ones((num_users, num_movies))

# Fill this

rmse_movie_mean_test = evaluate_rmse(rating_test, rating_movie_mean)
print(f"RMSE - movie_mean: {rmse_movie_mean_test:.6f}")

# --------------------------------------------------------------------
# Problem 1-3: Prediction with mean and user_mean and movie_mean
# --------------------------------------------------------------------
rating_user_movie_mean = mean * np.ones((num_users, num_movies))

# Fill this

rmse_user_movie_mean_test = evaluate_rmse(rating_test, rating_movie_mean)
print(f"RMSE - user_movie_mean: {rmse_user_movie_mean_test:.6f}")


# -------------------------------------------------------------------- 
# Problem 2: Recommendation with the SVD
# --------------------------------------------------------------------
def truncated_SVD(A, k):
    """ Perform the truncated SVD to A with k singular values
        Caution: np.linalg.svd returns 1-d array for singular values
        So, if you want to use it as a matrix you need to the np.diag function. Also, 
        full_matrices=False option makes the np.linalg.svd return the reduced form of SVD
    Args:
        A: original matrix (numpy array)
        k: the number of singular values (int)
    Returns:
        Ak: the rank-k matrix approxmiate A (numpy array)
    """
    
    # Fill this
     
    return Ak

# settings
k = 20           # k is the number of singular values
iters = 2        # iters is the maximum number of iterations

# initialize rating matrix
rating_svd_norm = np.zeros((num_users, num_movies))
rating_svd_norm = fill_rating(rating_svd_norm, rating_train_norm)

# main loop
for i in range(iters):
    # low-rank approximation by SVD    
    rating_svd_norm_new = truncated_SVD(rating_svd_norm, k)
    
    # update the ratining matrix
    rating_svd_norm = fill_rating(rating_svd_norm_new, rating_train_norm)

# evaluate the performance of algorithm
rating_svd = add_mean_values(rating_svd_norm, mean, user_mean, movie_mean)
rmse_svd_test = evaluate_rmse(rating_test, rating_svd)

print(f"RMSE - SVD: {rmse_svd_test:.6f}")



# -------------------------------------------------------------------- 
# Problem 3: Recommendation with the ALS
# --------------------------------------------------------------------

# settings
k = 20          # k is the dimension of latent space (= the number of singular values)
iters1 = 2      # iters1 is the maximum number of outer iterations (for iterative matrix completion)
iters2 = 10     # iters2 is the maximum number of inner iterations (for alternating least squares)
alpha = 0.1     # alpha is the hyperparamter to control the contribution of regularization (= lambda in Ridge regression formular)

# initialize matrices
Q = np.random.normal(0.0, 0.01, size=(num_users, k))   # Q is the latent factor matrix for users
P = np.random.normal(0.0, 0.01, size=(num_movies, k))  # P is the latent factor matrix for movies
Ik = np.identity(k)                                    # Ik is the k by k identity matrix for ridge regression
rating_als_norm = np.zeros((num_users, num_movies))
rating_als_norm = fill_rating(rating_als_norm, rating_train_norm)

# main loop
for i1 in range(iters1):
    for i2 in range(iters2):
        
        # Fill this (update Q and P)
        
        rating_als = np.matmul(Q, P.transpose())
        rating_als = add_mean_values(rating_als, mean, user_mean, movie_mean)
        rmse_als = evaluate_rmse(rating_train, rating_als)
        print(f"RMSE - ALS on Train dataset ({i1+1}-{i2+1}): {rmse_als:.6f}")

    rating_als_norm = np.matmul(Q, P.transpose())
    rating_als_norm = fill_rating(rating_als_norm, rating_train_norm)

# evaluate the performance of algorithm
rating_als = add_mean_values(rating_als_norm, mean, user_mean, movie_mean)
rmse_svd_test = evaluate_rmse(rating_test, rating_als)

print(f"RMSE - ALS: {rmse_svd_test:.6f}")


# -------------------------------------------------------------------- 
# Problem 4: Recommendation with the SGD
# --------------------------------------------------------------------
def learning_rate_schedule(epoch):
    if epoch < 5:
        learning_rate = 0.1
    elif epoch < 10:
        learning_rate = 0.01
    else:
        learning_rate = 0.001
    return learning_rate

# settings
k = 20          # k is the dimension of latent space (= the number of singular values)
epochs = 15     # epochs is the maximum number of repeatition over the entire training set
alpha = 0.05     # alpha is the hyperparamter to control the contribution of regularization (= lambda in Ridge regression formular)

# initialize matrices
Q = np.random.normal(0.0, 0.01, size=(num_users, k))      # Q is the latent factor matrix for users
P = np.random.normal(0.0, 0.01, size=(num_movies, k))     # P is the latent factor matrix for movies
rating_sgd_norm = np.zeros((num_users, num_movies))
rating_sgd_norm = fill_rating(rating_sgd_norm, rating_train_norm)

# main loop
for epoch in range(epochs):
    # learning rate scheduling
    learning_rate = learning_rate_schedule(epoch)

    # stocastic gradient descent over the entire training set
    for i in range(len(rating_train)):
        u = rating_train["user_id"][i]
        m = rating_train["movie_id"][i]
        r_true = rating_train["rating"][i]
        r_pred = mean + user_mean[u] + movie_mean[m] + np.dot(Q[u, :], P[m, :])
        r_diff = r_true - r_pred
                
        # Fill this (update Q and P)
        
    rating_sgd_norm = np.matmul(Q, P.transpose())
    rating_sgd = add_mean_values(rating_sgd_norm, mean, user_mean, movie_mean)
    rmse_sgd = evaluate_rmse(rating_train, rating_sgd)
    print(f"RMSE - SGD on Train dataset (epoch: {epoch+1}): {rmse_sgd:.6f}")

# evaluate the performance of algorithm
rmse_sgd = evaluate_rmse(rating_test, rating_sgd)
print(f"RMSE - SGD: {rmse_sgd:.6f}")
