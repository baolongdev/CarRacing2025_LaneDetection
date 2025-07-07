import numpy as np
import cv2


class BirdEyeTransformer:
    def __init__(self, src_points=None, dst_points=None):
        """
        Nếu không cung cấp, dùng giá trị mặc định phù hợp ảnh 1280x720
        """
        self.src = src_points
        self.dst = dst_points
        self.M = None
        self.Minv = None

    def compute_transform(self, img_shape):
        h, w = img_shape[:2]

        # Nếu chưa có src/dst, dùng mặc định cho ảnh 1280x720
        if self.src is None:
            self.src = np.float32([
                [w - 10, h],  # bottom right lane
                [0 + 10, h],  # bottom left lane
                [w // 2 - 40, h // 2 + 15],  # top left lane
                [w // 2 + 40, h // 2 + 15]   # top right lane
            ])

        if self.dst is None:
            self.dst = np.float32([
                [w, h],
                [0, h],
                [0, 0],
                [w, 0]
            ])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img, verbose=False):
        """
        Biến đổi phối cảnh từ ảnh gốc -> bird's eye view
        """
        if self.M is None:
            self.compute_transform(img.shape)

        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)

        if verbose:
            img_marked = img.copy()
            for pt in self.src:
                cv2.circle(img_marked, tuple(np.int32(pt)), 5, (0, 255, 0), -1)
            for pt in self.dst:
                cv2.circle(warped, tuple(np.int32(pt)), 5, (0, 0, 255), -1)

            cv2.imshow("Original with ROI", img_marked)
            cv2.imshow("Bird Eye View", warped)

        return warped

    def unwarp(self, img):
        """
        Biến đổi từ bird's eye view về góc nhìn camera gốc
        """
        if self.Minv is None:
            raise ValueError("Minv is not computed. Call warp() first.")

        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.Minv, (w, h), flags=cv2.INTER_LINEAR)
