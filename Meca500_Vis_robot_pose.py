import sys
import os
import time
import threading
import json
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import pyzed.sl as sl
import mecademicpy.robot as mdr

# -------------------------------------------------------------------
# Helper functions (forward kinematics, projection)
# -------------------------------------------------------------------

def get_dh_matrix(a, d, alpha, theta):
    alpha_rad = np.deg2rad(alpha)
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)*np.cos(alpha_rad),  np.sin(theta_rad)*np.sin(alpha_rad), a*np.cos(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)*np.cos(alpha_rad), -np.cos(theta_rad)*np.sin(alpha_rad), a*np.sin(theta_rad)],
        [0,                  np.sin(alpha_rad),                 np.cos(alpha_rad),                d],
        [0,                  0,                                0,                                1]
    ], dtype=np.float32)


def forward_kinematics(joint_angles):
    dh_params = [
        {'alpha': -90, 'a': 0,     'd': 0.135, 'theta_offset': 0},
        {'alpha': 0,   'a': 0.135, 'd': 0,     'theta_offset': -90},
        {'alpha': -90, 'a': 0.038, 'd': 0,     'theta_offset': 0},
        {'alpha': 90,  'a': 0,     'd': 0.120, 'theta_offset': 0},
        {'alpha': -90, 'a': 0,     'd': 0,     'theta_offset': 0},
        {'alpha': 0,   'a': 0,     'd': 0.070, 'theta_offset': 0}
    ]
    # base correction
    rot = R.from_euler('zx', [90,180], degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = rot

    coords = [np.array([0,0,0,1],dtype=np.float32)]
    for i, angle in enumerate(joint_angles):
        p = dh_params[i]
        theta = angle + p['theta_offset']
        T = T @ get_dh_matrix(p['a'], p['d'], p['alpha'], theta)
        coords.append(T @ np.array([0,0,0,1],dtype=np.float32))
    return np.vstack(coords)[:,:3]


def project_to_image(joints_3d, rvec, tvec, K, dist):
    # Rodrigues expects rvec in radians
    R_mat, _ = cv2.Rodrigues(rvec)
    pts_cam = (R_mat @ joints_3d.T).T + tvec.reshape(1,3)
    img_pts, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    return img_pts.reshape(-1,2)

# -------------------------------------------------------------------
# ArUco detection
# -------------------------------------------------------------------

def detect_aruco(frame, K, dist, marker_length=0.05):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        return None, None
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
    # take first marker
    return rvec[0].reshape(3), tvec[0].reshape(3)

# -------------------------------------------------------------------
# Real-time visualization per camera
# -------------------------------------------------------------------

def visualize_camera(camera_serial, calib_path, window_name):
    # load calibration
    calib = json.load(open(calib_path, 'r'))
    K = np.array(calib['camera_matrix'],dtype=np.float32)
    dist = np.array(calib['distortion_coeffs'],dtype=np.float32)

    # open Zed
    zed = sl.Camera()
    init = sl.InitParameters(); init.set_from_serial_number(camera_serial)
    zed.open(init)
    runtime = sl.RuntimeParameters()

    # init robot
    robot = mdr.Robot()
    robot.Connect(address="192.168.0.100", enable_synchronous_mode=False)
    robot.ActivateAndHome()
    robot.WaitHomed()
    print(f"Robot homed; starting visualization for camera {camera_serial}")

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        img = sl.Mat(); zed.retrieve_image(img, sl.VIEW.LEFT)  # or RIGHT/TOP based on serial
        frame = img.get_data()[:,:, :3]

        # detect pose
        ar_r, ar_t = detect_aruco(frame, K, dist)
        if ar_r is not None:
            joints = robot.GetJoints()
            coords3d = forward_kinematics(joints)
            pts2d = project_to_image(coords3d, ar_r, ar_t, K, dist).astype(int)
            # draw skeleton
            for i in range(len(pts2d)-1):
                cv2.line(frame, tuple(pts2d[i]), tuple(pts2d[i+1]), (255,0,255), 2)
            # draw joints
            for idx, p in enumerate(pts2d):
                cv2.circle(frame, tuple(p), 5, (0,255,255), -1)
                cv2.putText(frame, f"J{idx}", (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
            # draw axes
            origin = pts2d[0]; x_e = pts2d[1]; y_e = pts2d[2]; z_e = pts2d[3]
            cv2.line(frame, tuple(origin), tuple(x_e), (0,0,255),3); cv2.putText(frame, 'X', tuple(x_e), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.line(frame, tuple(origin), tuple(y_e), (0,255,0),3); cv2.putText(frame, 'Y', tuple(y_e), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            cv2.line(frame, tuple(origin), tuple(z_e), (255,0,0),3); cv2.putText(frame, 'Z', tuple(z_e), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()
    robot.DeactivateRobot(); robot.Disconnect()
    cv2.destroyWindow(window_name)

# -------------------------------------------------------------------
# Main: launch visualization threads
# -------------------------------------------------------------------
if __name__ == '__main__':
    # map serial to calibration file and name
    cams = {
        41182735: ('./calib/front_41182735_leftcam_calib.json','Front View'),
        49429257: ('./calib/right_49429257_rightcam_calib.json','Right View'),
        44377151: ('./calib/left_44377151_leftcam_calib.json','Left View'),
        49045152: ('./calib/top_49045152_topcam_calib.json','Top View'),
    }
    threads = []
    for serial, (calib, name) in cams.items():
        t = threading.Thread(target=visualize_camera, args=(serial, calib, name), daemon=True)
        threads.append(t); t.start()
    print("Press 'q' in any window to exit.")
    for t in threads: t.join()
