import cv2
import numpy as np
import glob
import pickle
import os
import os.path as path


class CameraCalibrator:
    def __init__(self, calib_dir='./models/camera_cal', cache_file='./models/camera_cal/calibration_data.pickle', chessboard_size=(9, 6)):
        self.calib_dir = calib_dir
        self.cache_file = cache_file
        self.chessboard_size = chessboard_size
        self.mtx = None
        self.dist = None
        self.ret = None
        self._load_or_calibrate()

    def _load_or_calibrate(self):
        if path.exists(self.cache_file):
            print("[INFO] Loading cached camera calibration...")
            with open(self.cache_file, 'rb') as f:
                self.ret, self.mtx, self.dist, _, _ = pickle.load(f)
        else:
            print("[INFO] Computing camera calibration from chessboard images...")
            self.ret, self.mtx, self.dist, rvecs, tvecs = self._calibrate()
            os.makedirs(path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump((self.ret, self.mtx, self.dist, rvecs, tvecs), f)
            print("[INFO] Calibration completed and saved to cache.")

    def _calibrate(self):
        objp = np.zeros((self.chessboard_size[1] * self.chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        images = glob.glob(path.join(self.calib_dir, 'calibration*.jpg'))
        if not images:
            raise FileNotFoundError(f"No images found in {self.calib_dir}")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pattern_found, corners = cv2.findChessboardCorners(gray, self.chessboard_size)

            if pattern_found:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

    def undistort(self, frame, verbose=False):
        if self.mtx is None or self.dist is None:
            raise ValueError("Camera calibration data not available.")

        undistorted = cv2.undistort(frame, self.mtx, self.dist, newCameraMatrix=self.mtx)

        if verbose:
            cv2.imshow("Original", frame)
            cv2.imshow("Undistorted", undistorted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return undistorted
