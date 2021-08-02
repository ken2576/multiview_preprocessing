'''Calibrate camera intrinsics given checkerboard videos (performed individually)
'''
import argparse
import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_cam(video_path, frame_interval=120, total_img_count=100):
    # Set up calibration flags
    CHECKERBOARD = (13,16)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW#cv2.fisheye.CALIB_CHECK_COND

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = 0

    while(1):
        if frame > total_frame or len(imgpoints) > total_img_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        frame += frame_interval
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Img', img)
            cv2.waitKey(1)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print("RMS: ", rms)

    return K, D

def calibrate_folder(folder, output='cam_params.pkl'):
    videos = sorted(glob.glob(os.path.join(folder, '*.mp4')))

    cams = []
    for video in videos:
        K, D = calibrate_cam(video)
        cam = {
            'K': K, # intrinsics
            'D': D  # distortion coefficients
        }
        cams.append(cam)

    with open(output, 'wb') as handle:
        pickle.dump(cams, handle)

def check_calibration():
    with open('cam_params.pkl', 'rb') as handle:
        b = pickle.load(handle)
        print(b)

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('root_dir', type=str,
                        help='root directory of video folder')
    parser.add_argument('--output', type=str,
                        help='output filename')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    calibrate_folder(args.root_dir, args.output)