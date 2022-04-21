import numpy as np

# A is a matrix of size 100 by 2
A = np.random.normal(0.0, 1.0, size=(100, 40))

# The full form of SVD
U, Sigma, VT = np.linalg.svd(A)
print("The full form of SVD: ", U.shape, Sigma.shape, VT.shape)

# The reduced form of SVD
U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
print("The reduced form of SVD: ", U.shape, Sigma.shape, VT.shape)

# np.diag can be used to generate a matrix for singular values
Sigma = np.diag(Sigma)
print("The reduced form of SVD: ", U.shape, Sigma.shape, VT.shape)

# Reconstructed A is same to the original A 
A_hat = np.matmul(np.matmul(U, Sigma), VT)
print("Original A[:5, :5]: \n", A[:5, :5])
print("Reconstructured A[:5, :5]: \n", A_hat[:5, :5])

# V matrix is the transpose of VT
V = VT.transpose()
print("V[:5, :5]: \n", V[:5, :5])
print("VT[:5, :5]: \n", VT[:5, :5])
