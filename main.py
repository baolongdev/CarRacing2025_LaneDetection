import cv2
from models import Controller, SimulatorClient
from pipeline import BirdEyeTransformer, ImageBinarizer, Line, get_fits_by_sliding_windows, draw_back_onto_the_road


class CarRacingApp:
    def __init__(self) -> None:
        self.client = SimulatorClient()
        self.binarizer = ImageBinarizer()
        self.birdeye = BirdEyeTransformer()
        self.controller = Controller()

        self.line_lt = Line(buffer_len=10)
        self.line_rt = Line(buffer_len=10)

        self.running = False

    def process_frame(self, frame):
        # Nhị phân hóa ảnh
        img_binary = self.binarizer.binarize(frame)
        # Chuyển sang bird's eye view
        img_birdeye = self.birdeye.warp(img_binary.astype('uint8') * 255)
    
        # Tìm làn đường bằng sliding windows
        self.line_lt, self.line_rt, _ = get_fits_by_sliding_windows(
            img_birdeye, self.line_lt, self.line_rt, n_windows=7, verbose=True
        )

        # Vẽ kết quả lên ảnh gốc
        result = draw_back_onto_the_road(
            Minv=self.birdeye.Minv,
            line_lt=self.line_lt,
            line_rt=self.line_rt,
            output_shape=frame.shape,  # hoặc (720, 1280, 3)
            keep_state=True
        )
        # steer, accel, mpc_x, mpc_y = self.controller.compute_control(
        #     self.line_lt, self.line_rt, self.client.state
        # )
        # result = self.mpc.draw_mpc_path_on_frame(result, mpc_x, mpc_y, color=(0, 255, 0))
        # self.client.send_control(steer, accel)
        cv2.imshow("Lane Detection", result)
        overlay = cv2.addWeighted(frame, 1.0, result, 0.5, 0)
        cv2.imshow("Lane Detection", overlay)

    def run(self) -> None:
        self.running = True
        try:
            while self.running:
                self.client.update(get_raw=True)
                frame = self.client.raw_image
                
                if frame is not None:
                    self.process_frame(frame)

                print(self.client.state)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            self.shutdown()
            cv2.destroyAllWindows()

    def shutdown(self) -> None:
        if self.client:
            print("[INFO] Shutting down CarRacingApp...")
            self.client.close()
        self.running = False

    def __del__(self):
        self.shutdown()


if __name__ == "__main__":
    app = CarRacingApp()
    app.run()