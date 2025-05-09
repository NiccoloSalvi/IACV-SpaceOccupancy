import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

def calibrate_camera(
    image_paths, board_size=(9, 6), square_size=1.0
):
    """
    Calibrates a camera given a list of checkerboard images.

    Args:
        image_paths (list of Path): Paths to calibration images.
        board_size (tuple): Number of inner corners per chessboard row and column (cols, rows).
        square_size (float): Size of a square in your defined unit (e.g. mm).

    Returns:
        ret: RMS re-projection error.
        camera_matrix: Intrinsic matrix K.
        dist_coeffs: Distortion coefficients.
        rvecs: Rotation vectors.
        tvecs: Translation vectors.
    """
    # Prepare object points (0,0,0), (1,0,0), ..., multiplied by square_size
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3-D points in real world space
    imgpoints = []  # 2-D points in image plane.

    for img_path in image_paths:
        img = cv.imread(str(img_path))
        if img is None:
            print(f"Warning: unable to load {img_path}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, corners = cv.findChessboardCorners(gray, board_size,
                                                 cv.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv.CALIB_CB_FAST_CHECK +
                                                 cv.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"Chessboard not found in {img_path}")
            continue

        # Refine corner locations
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners)
        objpoints.append(objp)

        # Optionally draw and display
        cv.drawChessboardCorners(img, board_size, corners, found)

        # resize for display
        h, w = img.shape[:2]
        disp_w, disp_h = int(w * 0.25), int(h * 0.25)
        resized = cv.resize(img, (disp_w, disp_h), interpolation=cv.INTER_AREA)
        cv.imshow('Corners', resized)
        cv.setWindowTitle('Corners', f"Chessboard corners in {img_path.name}")
        cv.moveWindow('Corners', 100, 100)

        # cv.imshow('Corners', img)
        cv.waitKey(0)

    cv.destroyAllWindows()

    if len(objpoints) < 5:
        raise RuntimeError("Not enough valid calibration images (found <5 valid views)")

    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("RMS re-projection error:", ret)
    print("Camera matrix (K):\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calibrate camera from chessboard images.'
    )
    parser.add_argument(
        '--images', type=Path, required=True,
        help='Directory containing calibration images.'
    )
    parser.add_argument(
        '--cols', type=int, default=9,
        help='Number of inner corners per chessboard row.'
    )
    parser.add_argument(
        '--rows', type=int, default=6,
        help='Number of inner corners per chessboard column.'
    )
    parser.add_argument(
        '--square-size', type=float, default=25.0,
        help='Size of one square in your chosen unit (e.g., millimeters).'
    )
    args = parser.parse_args()

    # 1. Work at full resolution if you can
    # img = cv.imread("checkboardFrames_2/frame_06.png")
    # h, w = img.shape[:2]
    # # gray = cv.cvtColor(img_small, …)  # avoid downscaling

    # # 2. Enhance contrast
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # # a) simple histogram equalization
    # gray = cv.equalizeHist(gray)
    # # b) (even better) CLAHE
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)

    # # 3. Denoise
    # gray = cv.GaussianBlur(gray, (5,5), 0)
    # # or
    # # gray = cv.bilateralFilter(gray, 9, 75, 75)

    # # 4. (Optional) morphological open/close to clean residuals
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    # gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

    # # 5. Try the SB detector first (more robust)
    # sb_flags = (
    #     cv.CALIB_CB_EXHAUSTIVE
    # | cv.CALIB_CB_ACCURACY
    # | cv.CALIB_CB_NORMALIZE_IMAGE
    # )
    # found, corners = cv.findChessboardCornersSB(
    #     gray, (8,6), flags=sb_flags
    # )

    # # 6. Fallback to the legacy method if SB misses
    # if not found:
    #     legacy_flags = (
    #         cv.CALIB_CB_ADAPTIVE_THRESH
    #     | cv.CALIB_CB_NORMALIZE_IMAGE
    #     )
    #     found, corners = cv.findChessboardCorners(
    #         gray, (8,6), flags=legacy_flags
    #     )

    # print("Found:", found)
    # if found:
    #     # sub‐pixel refine (important for calibration!)
    #     term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #     corners = cv.cornerSubPix(gray, corners, (5,5), (-1,-1), term)
    #     cv.drawChessboardCorners(img, (8,6), corners, found)


    # 7. Show the result
    # resize for display
    # disp_w, disp_h = int(w * 0.25), int(h * 0.25)
    # resized = cv.resize(img, (disp_w, disp_h), interpolation=cv.INTER_AREA)
    # cv.imshow('Corners', resized)
    # # cv.imshow("prep + detection", img)
    # cv.waitKey(0)

    # Gather image files
    img_dir = Path(args.images)
    images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    if len(images) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    calibrate_camera(
        images,
        board_size=(args.cols, args.rows),
        square_size=args.square_size
    )
