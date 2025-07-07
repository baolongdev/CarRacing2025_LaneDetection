from models.PIDController import PID
from models.KalmanFilter import KalmanFilter
from models.MPC import MPC
import numpy as np

class Controller:
    def __init__(self):
        # PID điều khiển heading
        self.pid = PID()
        self.pid.init(Kp=0.1, Ki=0.001, Kd=1.0)

        # Kalman filter để lọc state (nếu cần)
        self.kf = None  # có thể khởi tạo sau với state thực tế

        # MPC
        self.mpc = MPC()

    def compute_control(self, line_lt, line_rt, state_dict):
        """
        Nhận vào thông tin làn đường (line_lt, line_rt) và state từ simulator.
        Trả về góc lái và tốc độ (steer, accel).
        """
        # Tính offset từ trung tâm làn đường
        img_center = 1280 // 2
        left_fit = line_lt.average_fit
        right_fit = line_rt.average_fit

        ploty = np.linspace(0, 719, num=720)
        left_x = left_fit[0] * 719**2 + left_fit[1] * 719 + left_fit[2]
        right_x = right_fit[0] * 719**2 + right_fit[1] * 719 + right_fit[2]
        lane_center = (left_x + right_x) / 2.0
        offset_pixels = img_center - lane_center
        offset_meters = offset_pixels * 3.7 / 700  # nếu cần chuyển đổi đơn vị

        # PID control dựa trên offset
        self.pid.update_error(offset_meters)
        steer_pid = self.pid.total_error()

        # MPC control
        # Chuẩn bị state và fit lại đường cong giữa làn
        mid_fit = (left_fit + right_fit) / 2.0
        coeffs = mid_fit
        x = 0
        y = 0
        psi = 0
        v = state_dict.get("Speed", 0)
        cte = np.polyval(coeffs, 0)
        epsi = -np.arctan(coeffs[1])  # đạo hàm bậc 1 tại x=0

        state = np.array([x, y, psi, v, cte, epsi])
        steer_mpc, accel_mpc, mpc_x, mpc_y = self.mpc.solve(state, coeffs)

        return steer_mpc, accel_mpc, mpc_x, mpc_y
