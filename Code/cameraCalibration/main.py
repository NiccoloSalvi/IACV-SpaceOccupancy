import numpy as np
import cv2 as cv
import os

def calibrate_camera(image_paths, board_size=(9, 6), square_size=1.0):
    # Prepare object points (0,0,0), (1,0,0), ..., multiplied by square_size
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3-D points in real world space
    imgpoints = []  # 2-D points in image plane

    # for every image in the directory
    image_files = sorted([f for f in os.listdir(image_paths) if f.endswith('.png')])
    image_paths = [os.path.join(image_paths, f) for f in image_files]
    for img_path in image_paths:
        img = cv.imread(img_path)
        if img is None:
            print(f"Warning: unable to load {img_path}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, corners = cv.findChessboardCorners(gray, board_size, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"Chessboard not found in {img_path}")
            continue

        # Refine corner locations
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners)
        objpoints.append(objp)

        # Optionally draw and display
        drawnImage = cv.drawChessboardCorners(img, board_size, corners, found)

        # resize for display
        h, w = img.shape[:2]
        disp_w, disp_h = int(w * 0.25), int(h * 0.25)
        resized = cv.resize(drawnImage, (disp_w, disp_h))

        save_path = os.path.join('calibration_results', os.path.basename(img_path))
        if not os.path.exists('calibration_results'):
            os.makedirs('calibration_results')
        print(f"Saving processed image to {save_path}")
        cv.imwrite(os.path.join('calibration_results', os.path.basename(img_path)), resized)

    cv.destroyAllWindows()

    if len(objpoints) < 5:
        raise RuntimeError("Not enough valid calibration images (found <5 valid views)")

    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("RMS re-projection error:", ret)
    print("Camera matrix (K):\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

if __name__ == '__main__':
    path_images = os.path.join(os.getcwd(), 'cameraCalibration', 'checkboard_3')
    cols = 8
    rows = 6
    square_size = 24.5  # Size of one square in mm

    calibrate_camera(path_images, board_size=(cols, rows), square_size=square_size)