import numpy as np

class RLS:
    """Implements Recursive Least Squares filter with a vector as input and another vector as output.
    Shapes:
      X: M x 1
      Y: N x 1 
      W: M x N
      P: M x M
      K: M x 1

    For our application:
    The idea is to forecast a vector Y(n) corresponding with entries corresponding to different sites.
    The forecast is done from a vector of inputs X(n) using a matrix of weights W(n).
    Let L be the number of sites, N be the dimensionality of the input vector of site
    """

    def __init__(self, M, N, forgetting_factor=0.99, uncertainty=10):
        self.reset(M, N, forgetting_factor, uncertainty)

    def reset(self, M, N, forgetting_factor=0.99, uncertainty=10):
        """ Initialize coefficients and uncertainty """
        self.W = np.random.randn(M, N)  # assumed to be MxN
        self.P = np.eye(N) * uncertainty  # assumed to be NxN
        self.ff = forgetting_factor
        self.check_shapes()

    def check_shapes(self):
        assert (
            self.X.shape[0]
            == self.W.shape[0]
            == self.P.shape[0]
            == self.P.shape[1]
            == self.K.shape[0]
        )
        assert (
            1
            == self.X.shape[1]
            == self.Y.shape[1]
            == self.K.shape[1]
        )
        assert (
            2
            == self.X.ndim
            == self.Y.ndim
            == self.P.ndim
            == self.K.ndim
        )
        assert type(self.ff) == float

    def set_weights(self, W):
        """ Set as weights the function argument

        Args:
            W (numpy.ndarray): is desirable that they are well shaped
        """
        self.W = W

    def predict(self, X_in):
        """ Makes the prediction W.T @ X_in

        Args:
            X_in (numpy.ndarray): input vector, assumed to be N x 1

        Returns:
            numpy.ndarray: prediction of shape M x 1
        """
        return self.W.T @ X_in

    def update(self, X, Y):
        """ Updates according to the equations in the overleaf 'general_recursive_least_squares'
        """
        self.K = self.P @ X / (self.ff + X.T @ self.P @ X)
        self.P = (self.P - self.K @ X.T @ self.P) / self.ff
        self.W = self.W + self.K @ (Y.T - X.T @ self.W)

