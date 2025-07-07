class PID:
    def __init__(self):
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        self.error_proportional = 0.0
        self.error_integral = 0.0
        self.error_derivative = 0.0

    def init(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def update_error(self, cte: float):
        self.error_integral += cte
        self.error_derivative = cte - self.error_proportional
        self.error_proportional = cte

    def total_error(self) -> float:
        return -(self.Kp * self.error_proportional +
                 self.Ki * self.error_integral +
                 self.Kd * self.error_derivative)
