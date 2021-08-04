'''Calibrate camera intrinsics given checkerboard videos (performed individually) and selected frame folder
'''
import argparse
import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_cam(video_path, frames=[], show_img=False):
    # Set up calibration flags
    CHECKERBOARD = (13,16)
    SQUARE_SIZE = 40.0
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_CHECK_COND

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(video_path)

    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
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
            imgpoints.append(corners2)

            if show_img:
                cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                cv2.imshow('Img', img)
                cv2.waitKey(1)

    rms, mtx, dist, rvecs, tvecs = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            None,
            None,
            None,
            None,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(mtx.tolist()) + ")")
    print("D=np.array(" + str(dist.tolist()) + ")")
    print("RMS: ", rms)

    return mtx, dist

def calibrate_folder(folder, frame_folder, output='cam_params.pkl', img_count=100):
    def get_frames(paths):
        frames = []
        for path in paths:
            curr_id = os.path.split(path)[-1]
            curr_id = int(''.join(x for x in curr_id if x.isdigit()))
            frames.append(curr_id)
        return frames

    videos = sorted(glob.glob(os.path.join(folder, '*.mp4')))
    
    imgs = sorted(glob.glob(os.path.join(frame_folder, '*.png')), key=os.path.getmtime)
    frames = get_frames(imgs)

    if len(frames) > img_count:
        frames = np.random.choice(frames, img_count, replace=False)

    cams = []
    for video in videos:
        K, D = calibrate_cam(video, frames)
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
    parser.add_argument('--frame_folder', type=str,
                        help='directory of predetermined frames')
    parser.add_argument('--output', type=str,
                        help='output filename')
    parser.add_argument('--img_count', type=int,
                        help='max image count to use', default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    calibrate_folder(args.root_dir, args.frame_folder, args.output, args.img_count)