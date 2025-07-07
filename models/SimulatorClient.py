from client_lib import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket


class SimulatorClient:
    def __init__(self):
        self.state = None       # Current simulation state
        self.raw_image = None   # Raw camera image
        self.seg_image = None   # Segmentation image

        self.speed = 0.0        # Current speed command
        self.angle = 0.0        # Current steering command

    def update(self, get_raw: bool = False, get_seg: bool = False) -> None:
        """
        Update the simulation state and optionally retrieve raw and segmentation images.
        """
        self.state = GetStatus()
        self.raw_image = GetRaw() if get_raw else None
        self.seg_image = GetSeg() if get_seg else None

    def send_control(self, steer: float, accel: float) -> None:
        """
        Send control commands to the simulator.

        Args:
            steer (float): Steering angle in range [-25, 25] degrees.
            accel (float): Acceleration or speed command in range [-100, 100].
        """
        self.angle = max(-25, min(25, steer))
        self.speed = max(-100, min(100, accel))
        AVControl(speed=self.speed, angle=self.angle)

    def close(self) -> None:
        """Close the socket connection to the simulation."""
        print("Closing socket...")
        CloseSocket()
