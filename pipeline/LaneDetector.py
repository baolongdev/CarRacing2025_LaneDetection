import numpy as np
import cv2
from config import Config


class Line:
    def __init__(self, buffer_len=10):
        self.detected = False
        self.last_fit_pixel = None
        self.last_fit_meter = None
        self.recent_fits_pixel = []
        self.recent_fits_meter = []
        self.radius_of_curvature = None
        self.all_x = None
        self.all_y = None
        self.buffer_len = buffer_len

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        self.detected = detected
        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(new_fit_pixel)
        self.recent_fits_pixel = self.recent_fits_pixel[-self.buffer_len:]

        self.recent_fits_meter.append(new_fit_meter)
        self.recent_fits_meter = self.recent_fits_meter[-2 * self.buffer_len:]

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        h, w, _ = mask.shape
        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel
        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])
        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])
    
    
    

def get_fits_by_sliding_windows(birdeye_binary, line_lt, line_rt, n_windows=9, verbose=False):
    height, width = birdeye_binary.shape
    histogram = np.sum(birdeye_binary[height // 2:-30, :], axis=0)
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(height / n_windows)
    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * Config.YM_PER_PIX, line_lt.all_x * Config.XM_PER_PIX, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * Config.YM_PER_PIX, line_rt.all_x * Config.XM_PER_PIX, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected)

    if verbose:
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit_pixel[0] * ploty**2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0] * ploty**2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        # Vẽ đường fitted line bằng cv2
        for i in range(len(ploty) - 1):
            pt1_l = (int(left_fitx[i]), int(ploty[i]))
            pt2_l = (int(left_fitx[i+1]), int(ploty[i+1]))
            pt1_r = (int(right_fitx[i]), int(ploty[i]))
            pt2_r = (int(right_fitx[i+1]), int(ploty[i+1]))

            cv2.line(out_img, pt1_l, pt2_l, (255, 255, 0), 2)   # Yellow for left line
            cv2.line(out_img, pt1_r, pt2_r, (255, 255, 0), 2)   # Yellow for right line

        # Hiển thị ảnh nhị phân và ảnh có fitted line
        birdeye_vis = cv2.cvtColor(birdeye_binary * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("BirdEye Binary", birdeye_vis)
        cv2.imshow("Fitted Lines", out_img)


    return line_lt, line_rt, out_img


def get_fits_by_previous_fits(birdeye_binary, line_lt, line_rt, verbose=False):
    height, width = birdeye_binary.shape
    left_fit = line_lt.last_fit_pixel
    right_fit = line_rt.last_fit_pixel

    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100

    left_inds = ((nonzero_x > (left_fit[0]*nonzero_y**2 + left_fit[1]*nonzero_y + left_fit[2] - margin)) &
                 (nonzero_x < (left_fit[0]*nonzero_y**2 + left_fit[1]*nonzero_y + left_fit[2] + margin)))
    right_inds = ((nonzero_x > (right_fit[0]*nonzero_y**2 + right_fit[1]*nonzero_y + right_fit[2] - margin)) &
                  (nonzero_x < (right_fit[0]*nonzero_y**2 + right_fit[1]*nonzero_y + right_fit[2] + margin)))

    line_lt.all_x, line_lt.all_y = nonzero_x[left_inds], nonzero_y[left_inds]
    line_rt.all_x, line_rt.all_y = nonzero_x[right_inds], nonzero_y[right_inds]

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * Config.YM_PER_PIX, line_lt.all_x * Config.xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * Config.YM_PER_PIX, line_rt.all_x * Config.xm_per_pix, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected)

    return line_lt, line_rt, None


def draw_back_onto_the_road(Minv, line_lt, line_rt, output_shape, keep_state):
    height, width = output_shape[:2]

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Tạo ảnh nền đen để vẽ polygon lane
    road_warp = np.zeros((height, width, 3), dtype=np.uint8)

    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    pts = np.vstack([pts_left, pts_right])

    # Vẽ vùng lane
    cv2.fillPoly(road_warp, [np.int32(pts)], (0, 255, 0))  # màu xanh lá

    # Vẽ lại các lane line
    line_mask = np.zeros_like(road_warp)
    line_mask = line_lt.draw(line_mask, color=(255, 0, 0), average=keep_state)
    line_mask = line_rt.draw(line_mask, color=(0, 0, 255), average=keep_state)

    lane_combined = cv2.addWeighted(road_warp, 1.0, line_mask, 1.0, 0)

    # Nếu cần trả về ở góc nhìn camera gốc
    road_dewarped = cv2.warpPerspective(lane_combined, Minv, (width, height))

    return road_dewarped
