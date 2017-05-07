import numpy.linalg.solve as solve
from numpy import transpose


# For the notations:
# See https://stanford.edu/class/ee363/lectures/kf.pdf slide 14=>16
# xt = A*xtm1 + V with V normally distributed N(mean_V,sigma_V)
# yt = C*xt + W with W normally distributed N(mean_W,sigma_W)
# if A,C meanV/W or sigmaV/W are time dependant then update them after invoking self.update()

class KalmanFilter:
    def __init__(self, prior_x, prior_sigma, A, C, sigma_V, sigma_W, mean_V=0, mean_W=0):
        self.x_tp1 = prior_x;
        self.sigma_tp1 = prior_sigma;
        self.A = A
        self.C = C
        self.sigma_V
        self.sigma_W
        self.mean_V = mean_V
        self.mean_W = mean_W
        self.x_t = 0
        self.sigma_t = 0

    ####################
    ### public method
    ####################

    def update(self, y):
        self._measurement_update(y)
        self._time_update()

    ####################
    ### private method
    ####################


    def _measurement_update(self, y):
        K = self.sigma_tp1 * transpose(self.C)
        normalization_matrix = self.C * K + self.sigma_V

        self.x_t = self.x_tp1 + K * solve(normalization_matrix, \
                                          y - self.C * self.x_tp1 - self.mean_W)
        self.sigma_t = self.sigma_tp1 - K * solve(normalization_matrix, transpose(K))

    def _time_update(self):
        self.x_tp1 = self.A * self.x_t + self.mean_V
        self.sigma_tp1 = self.A * self.sigma_t * transpose(self.A) + self.sigma_W

    ####################
    ### getter/setter
    ####################

    def get_A(self):
        return self.A

    def set_A(self, value):
        self.A = value;

    def get_C(self):
        return self.C

    def set_C(self, value):
        self.C = value;

    def get_V(self):
        return (self.mean_V, self.sigma_V)

    def set_V(self, mean, sigma):
        self.mean_V = mean;
        self.sigma_V = sigma

    def get_W(self):
        return (self.mean_W, self.sigma_W)

    def set_W(self, mean, sigma):
        self.mean_W = mean
        self.sigma_W = sigma

    ####################
    ### properties
    ####################

    A = property(get_A, set_A)
    C = property(get_C, set_C)
    V = property(get_V, set_V)
    W = property(get_W, set_W)
