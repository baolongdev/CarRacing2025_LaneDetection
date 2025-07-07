import numpy as np

class KalmanFilter:
    def __init__(self):
        self.x = None  # state vector
        self.P = None  # state covariance matrix
        self.F = None  # state transition matrix
        self.H = None  # measurement matrix
        self.R = None  # measurement covariance matrix
        self.Q = None  # process noise covariance matrix

    def init(self, x_in, P_in, F_in, H_in, R_in, Q_in):
        self.x = x_in.copy()
        self.P = P_in.copy()
        self.F = F_in.copy()
        self.H = H_in.copy()
        self.R = R_in.copy()
        self.Q = Q_in.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update_routine(self, y):
        Ht = self.H.T
        S = self.H @ self.P @ Ht + self.R
        K = self.P @ Ht @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def update(self, z):
        z_pred = self.H @ self.x
        y = z - z_pred
        self._update_routine(y)

    def update_ekf(self, z):
        px, py, vx, vy = self.x[0], self.x[1], self.x[2], self.x[3]

        rho = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        rho_dot = (px * vx + py * vy) / max(rho, 1e-4)

        z_pred = np.array([rho, phi, rho_dot])
        y = z - z_pred

        # Normalize angle
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

        self._update_routine(y)
