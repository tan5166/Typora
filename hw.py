import numpy as np
import matplotlib.pyplot as plt
import random

def generate_data(k):
  np.random.seed(k)
  y = np.random.choice([-1, 1])
  if y == 1:
    mean = [3, 2]
    covariance_matrix = [[0.4, 0], [0, 0.4]]
    x1, x2 = np.random.multivariate_normal(mean, covariance_matrix)
    x = np.array([1, x1, x2])
  else:
    mean = [5, 0]
    covariance_matrix = [[0.6, 0], [0, 0.6]]
    x1, x2 = np.random.multivariate_normal(mean, covariance_matrix)
    x = np.array([1, x1, x2])
  return x,y



training_N = 256
test_N = 4096
experiments_N = 128

x = np.zeros([training_N,3],dtype=np.float64)
y = np.zeros([training_N,1])

err_LIN = np.zeros([experiments_N,1])
# err_01 = np.zeros([experiments_N,1])
err_01 = np.zeros([experiments_N,1],dtype=np.float64)

err_check = np.zeros([experiments_N,1],dtype=np.float64)

for j in range(experiments_N):
  for i in range(training_N):
    x[i], y[i] = generate_data(i+j)
  x_pseudoinverse = np.linalg.pinv(x)
  w_LIN = x_pseudoinverse @ y
  err_LIN[j] = (np.linalg.norm(x @ w_LIN - y, ord=2) **2) / training_N
  #err_01[j] = (np.count_nonzero(np.sign(x @ w_LIN) - y)) / training_N
  err_01[j] = np.sum(x @ w_LIN)
  for k in range(training_N):
    err_check[j] += np.dot(w_LIN.T,x[k])
