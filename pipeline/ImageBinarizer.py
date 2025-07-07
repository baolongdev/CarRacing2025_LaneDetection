import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageBinarizer:
    def __init__(self,
                 sobel_kernel_size=9,
                 sobel_thresh=50,
                 white_thresh=250,
                 morph_kernel_size=(5, 5)):
        self.sobel_kernel_size = sobel_kernel_size
        self.sobel_thresh = sobel_thresh
        self.white_thresh = white_thresh
        self.morph_kernel = np.ones(morph_kernel_size, np.uint8)

    def thresh_frame_sobel(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel_size)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
        _, binary = cv2.threshold(sobel_mag, self.sobel_thresh, 1, cv2.THRESH_BINARY)
        return binary.astype(bool)

    def get_binary_from_equalized_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(eq, self.white_thresh, 255, cv2.THRESH_BINARY)
        return binary > 0

    def binarize(self, img, verbose=False):
        h, w = img.shape[:2]
        binary = np.zeros((h, w), dtype=bool)

        white_mask = self.get_binary_from_equalized_grayscale(img)
        sobel_mask = self.thresh_frame_sobel(img)

        binary |= white_mask
        binary |= sobel_mask

        # Morphological closing
        closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, self.morph_kernel)

        if verbose:
            self._visualize(img, white_mask, sobel_mask, binary, closing)

        return closing

    def _visualize(self, img, white_mask, sobel_mask, binary, closing):
        # Resize all images to same size for consistent display
        h, w = img.shape[:2]
        white_mask = cv2.resize(white_mask.astype(np.uint8) * 255, (w, h))
        sobel_mask = cv2.resize(sobel_mask.astype(np.uint8) * 255, (w, h))
        binary_vis = cv2.resize(binary.astype(np.uint8) * 255, (w, h))
        closing_vis = cv2.resize(closing.astype(np.uint8) * 255, (w, h))

        # Convert single-channel to BGR for better visualization in OpenCV
        white_mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        sobel_mask_bgr = cv2.cvtColor(sobel_mask, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
        closing_bgr = cv2.cvtColor(closing_vis, cv2.COLOR_GRAY2BGR)

        # Stack into 2 rows, 2 columns
        top_row = np.hstack((img, white_mask_bgr))
        bottom_row = np.hstack((sobel_mask_bgr, closing_bgr))
        combined = np.vstack((top_row, bottom_row))

        # Show using OpenCV
        cv2.imshow("Binarization Debug View", combined)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

