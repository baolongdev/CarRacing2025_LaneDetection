import numpy as np
from scipy.optimize import minimize


class MPC:
    def __init__(self, N=10, dt=0.1, Lf=2.67):
        self.N = N      # Prediction horizon
        self.dt = dt    # Time step duration
        self.Lf = Lf    # Distance between center of mass and front axle

        # Reference values
        self.ref_v = 60.0  # desired speed (mph)

    def _objective(self, vars, coeffs):
        """
        Cost function for MPC optimization.
        """
        cost = 0.0
        N = self.N

        x = vars[0:N]
        y = vars[N:2*N]
        psi = vars[2*N:3*N]
        v = vars[3*N:4*N]
        cte = vars[4*N:5*N]
        epsi = vars[5*N:6*N]
        delta = vars[6*N:6*N+N-1]
        a = vars[6*N+N-1:]

        for t in range(N):
            cost += 2000 * cte[t]**2
            cost += 2000 * epsi[t]**2
            cost += (v[t] - self.ref_v)**2

        for t in range(N - 1):
            cost += 10 * delta[t]**2
            cost += 10 * a[t]**2

        for t in range(N - 2):
            cost += 100 * (delta[t+1] - delta[t])**2
            cost += 10 * (a[t+1] - a[t])**2

        return cost

    def _constraints(self, vars, state, coeffs):
        """
        Constraints for vehicle kinematics.
        """
        N = self.N
        dt = self.dt
        Lf = self.Lf

        x0, y0, psi0, v0, cte0, epsi0 = state

        x = vars[0:N]
        y = vars[N:2*N]
        psi = vars[2*N:3*N]
        v = vars[3*N:4*N]
        cte = vars[4*N:5*N]
        epsi = vars[5*N:6*N]
        delta = vars[6*N:6*N+N-1]
        a = vars[6*N+N-1:]

        constraints = []

        constraints += [x[0] - x0]
        constraints += [y[0] - y0]
        constraints += [psi[0] - psi0]
        constraints += [v[0] - v0]
        constraints += [cte[0] - cte0]
        constraints += [epsi[0] - epsi0]

        for t in range(1, N):
            f_t = np.polyval(coeffs[::-1], x[t-1])
            f_prime_t = np.polyval(np.polyder(coeffs[::-1]), x[t-1])
            psides = np.arctan(f_prime_t)

            constraints += [x[t] - (x[t-1] + v[t-1] * np.cos(psi[t-1]) * dt)]
            constraints += [y[t] - (y[t-1] + v[t-1] * np.sin(psi[t-1]) * dt)]
            constraints += [psi[t] - (psi[t-1] + v[t-1] * delta[t-1] / Lf * dt)]
            constraints += [v[t] - (v[t-1] + a[t-1] * dt)]
            constraints += [cte[t] - ((f_t - y[t-1]) + v[t-1] * np.sin(epsi[t-1]) * dt)]
            constraints += [epsi[t] - ((psi[t-1] - psides) + v[t-1] * delta[t-1] / Lf * dt)]

        return np.array(constraints)

    def solve(self, state, coeffs):
        """
        Solve the MPC optimization problem.
        """
        N = self.N
        n_vars = N * 6 + (N - 1) * 2
        vars_init = np.zeros(n_vars)

        bounds = [(-1e3, 1e3)] * n_vars
        for i in range(6*N, 6*N + N - 1):  # delta bounds
            bounds[i] = (-0.436, 0.436)
        for i in range(6*N + N - 1, n_vars):  # acceleration bounds
            bounds[i] = (-1.0, 1.0)

        constraints = {
            'type': 'eq',
            'fun': lambda vars: self._constraints(vars, state, coeffs)
        }

        solution = minimize(
            lambda vars: self._objective(vars, coeffs),
            vars_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not solution.success:
            print("[WARN] MPC failed to find solution")

        opt_vars = solution.x
        steer = opt_vars[6*N] if solution.success else 0.0
        accel = opt_vars[6*N + N - 1] if solution.success else 0.0

        mpc_path_x = opt_vars[0:N]
        mpc_path_y = opt_vars[N:2*N]

        return steer, accel, mpc_path_x, mpc_path_y

    def draw_mpc_path_on_frame(self, frame, mpc_x, mpc_y, color=(0, 255, 0), radius=3):
        """
        Vẽ đường đi dự đoán của MPC lên ảnh đầu vào.

        Args:
            frame (np.ndarray): ảnh gốc (BGR)
            mpc_x (array-like): tọa độ x trong tọa độ ảnh (pixels)
            mpc_y (array-like): tọa độ y trong tọa độ ảnh (pixels)
            color (tuple): màu vẽ đường đi (mặc định xanh lá)
            radius (int): bán kính mỗi điểm
        Returns:
            frame_with_path (np.ndarray): ảnh đã vẽ đường MPC
        """
        frame_vis = frame.copy()
        h, w = frame.shape[:2]

        for x, y in zip(mpc_x, mpc_y):
            ix = int(x)
            iy = int(y)
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(frame_vis, (ix, iy), radius, color, -1)

        return frame_vis
