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
from queue import Queue

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
        {'alpha': 0,   'a': 0,     'd': 0.070, 'theta_offset': 0}, 
        {'alpha': 0,   'a': 0,     'd': 0.130, 'theta_offset': 0},
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
    R_mat, _ = cv2.Rodrigues(rvec)
    pts_cam = (R_mat @ joints_3d.T).T + tvec.reshape(1,3)
    img_pts, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    return img_pts.reshape(-1,2)

# -------------------------------------------------------------------
# ArUco detection
# -------------------------------------------------------------------

def detect_aruco(frame, K, dist, marker_length=0.05):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    params = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        return None, None
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
    return rvec[0].reshape(3), tvec[0].reshape(3)


def update_joints(robot, shared_joints, joint_lock):
    while True:
        joints = robot.GetJoints()
        joints.append(0.0)
        with joint_lock:
            shared_joints[:] = joints 
        time.sleep(0.05)


def robot_motion_loop(robot):
    poses = [
        (175, 123, 180, -150, 0, 150),
        (150, 150, 200, -100, 50, 100),
        (180, 70, 200, -120, 0, 120),
        (175, 123, 180, -150, 0, 150),
        (175, 136, 158, -150, 0, 150)
    ]
    while True:
        for pose in poses:
            robot.MovePose(*pose)
            robot.WaitIdle()
            time.sleep(0.05)

# -------------------------------------------------------------------
# Camera Frame Collector Thread
# -------------------------------------------------------------------
def camera_thread(serial, calib_path, shared_joints, joint_lock, frame_queue):
    # 카메라 초기화 (한 번만)
    zed = sl.Camera()
    init = sl.InitParameters(); init.set_from_serial_number(serial)
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera {serial}")
        return
    
    runtime = sl.RuntimeParameters()

    calib = json.load(open(calib_path, 'r'))
    K = np.array(calib['camera_matrix'], dtype=np.float32)
    dist = np.array(calib['distortion_coeffs'], dtype=np.float32)

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        img = sl.Mat(); zed.retrieve_image(img, sl.VIEW.LEFT)
        frame = img.get_data()[:, :, :3].copy()  # 복사본 생성

        # 아루코 마커 및 로봇 좌표 시각화
        ar_r, ar_t = detect_aruco(frame, K, dist)
        if ar_r is not None:
            with joint_lock:
                joints = list(shared_joints)
            print(f"joints: {joints}")
            coords3d = forward_kinematics(joints)
            pts2d = project_to_image(coords3d, ar_r, ar_t, K, dist).astype(int)

            for i in range(len(pts2d)-1):
                cv2.line(frame, tuple(pts2d[i]), tuple(pts2d[i+1]), (255,0,255), 2)
            for idx, p in enumerate(pts2d):
                cv2.circle(frame, tuple(p), 5, (0,255,255), -1)
                cv2.putText(frame, f"J{idx}", (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)

        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == '__main__':
    robot = mdr.Robot()
    robot.Connect(address="192.168.0.100", enable_synchronous_mode=False)
    robot.ActivateAndHome()
    robot.WaitHomed()

    robot.SetJointVel(10)  # 속도 2 deg/s
    print("Joint velocity set to 2 deg/s")

    print("Robot activated and homed successfully")

    shared_joints = [0.0] * 7
    joint_lock = threading.Lock()

    t_joint = threading.Thread(target=update_joints, args=(robot, shared_joints, joint_lock), daemon=True)
    t_joint.start()

    t_motion = threading.Thread(target=robot_motion_loop, args=(robot,), daemon=True)
    t_motion.start()
    cams = {
        41182735: ('./calib/front_41182735_leftcam_calib.json','Front View'),
        49429257: ('./calib/right_49429257_leftcam_calib.json','Right View'),
        44377151: ('./calib/left_44377151_leftcam_calib.json','Left View'),
        49045152: ('./calib/top_49045152_leftcam_calib.json','Top View'),
    }

    frame_queues = {}
    threads = []

    for serial, (calib_path, window_name) in cams.items():
        q = Queue(maxsize=1)
        frame_queues[window_name] = q
        t = threading.Thread(
            target=camera_thread,
            args=(serial, calib_path, shared_joints, joint_lock, q),
            daemon=True
        )
        threads.append(t)
        t.start()

    print("Press 'q' in any window to exit.")
    try:
        while True:
            for window_name, q in frame_queues.items():
                if not q.empty():
                    frame = q.get()
                    cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted.")

    cv2.destroyAllWindows()
    robot.DeactivateRobot()
    robot.Disconnect()


