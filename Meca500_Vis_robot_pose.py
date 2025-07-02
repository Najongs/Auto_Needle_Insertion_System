import sys
import os
import time
import threading
import json
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R, Slerp
import pyzed.sl as sl
import mecademicpy.robot as mdr
from queue import Queue

# -------------------------------------------------------------------
# Forward Kinematics / Projection
# -------------------------------------------------------------------

def get_dh_matrix(a, d, alpha, theta):
    alpha_rad = np.deg2rad(alpha)
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad) * np.cos(alpha_rad),  np.sin(theta_rad) * np.sin(alpha_rad), a * np.cos(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad) * np.cos(alpha_rad), -np.cos(theta_rad) * np.sin(alpha_rad), a * np.sin(theta_rad)],
        [0,                  np.sin(alpha_rad),                     np.cos(alpha_rad),                      d],
        [0,                  0,                                     0,                                      1]
    ], dtype=np.float32)


def forward_kinematics(joint_angles):
    """Compute 3D positions of robot joints using DH parameters."""
    dh_params = [
        {'alpha': -90, 'a': 0,     'd': 0.135, 'theta_offset': 0},
        {'alpha': 0,   'a': 0.135, 'd': 0,     'theta_offset': -90},
        {'alpha': -90, 'a': 0.038, 'd': 0,     'theta_offset': 0},
        {'alpha': 90,  'a': 0,     'd': 0.120, 'theta_offset': 0},
        {'alpha': -90, 'a': 0,     'd': 0,     'theta_offset': 0},
        {'alpha': 0,   'a': 0,     'd': 0.070, 'theta_offset': 0},
        {'alpha': 0,   'a': 0,     'd': 0.130, 'theta_offset': 0},
    ]

    joint_coords = [np.array([0, 0, 0])]
    T = np.eye(4)

    # Base frame correction: rotate 180° around X, then 90° around Z
    rot_x_180 = R.from_euler('x', 180, degrees=True)
    rot_z_90 = R.from_euler('z', 90, degrees=True)
    T[:3, :3] = (rot_z_90 * rot_x_180).as_matrix()

    base_point = np.array([[0], [0], [0], [1]])

    for i in range(7):
        p = dh_params[i]
        theta = joint_angles[i] + p['theta_offset']
        T_i = get_dh_matrix(p['a'], p['d'], p['alpha'], theta)
        T = T @ T_i
        joint_pos = T @ base_point
        joint_coords.append(joint_pos[:3, 0])

    return np.array(joint_coords, dtype=np.float32)


def project_to_image(joints_3d, rvec, tvec, K, dist):
    """Project 3D joints into 2D image using extrinsics and intrinsics."""
    R_mat, _ = cv2.Rodrigues(rvec)
    pts_cam = (R_mat @ joints_3d.T).T + tvec.reshape(1, 3)
    img_pts, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    return img_pts.reshape(-1, 2)

# -------------------------------------------------------------------
# Joint Updates / Robot Motions
# -------------------------------------------------------------------

def update_joints(robot, shared_joints, joint_lock):
    """Continuously update joint positions from robot."""
    while True:
        joints = robot.GetJoints()
        joints.append(0.0)  # 7th dummy joint for FK compatibility
        with joint_lock:
            shared_joints[:] = joints
        time.sleep(0.05)

def robot_motion_loop(robot):
    """Loop through predefined robot poses."""
    poses = [
        (175, 123, 180, -150, 0, 150),
        (150, 150, 200, -100, 50, 100),
        (180, 70,  200, -120, 0, 120),
        (175, 123, 180, -150, 0, 150),
        (175, 136, 158, -150, 0, 150),
    ]
    while True:
        for pose in poses:
            robot.MovePose(*pose)
            robot.WaitIdle()
            time.sleep(0.05)

# -------------------------------------------------------------------
# ArUco detection
# -------------------------------------------------------------------

marker_size = 0.05  # 5cm
marker_3d_edges = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# -------------------------------------------------------------------
# Camera Thread per ZED
# -------------------------------------------------------------------

def camera_thread(serial, calib_path, shared_joints, joint_lock, frame_queue):
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open camera {serial}")
        return
    runtime = sl.RuntimeParameters()

    # Calibration load
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    K = np.array(calib['camera_matrix'], dtype=np.float32)
    dist = np.array(calib['distortion_coeffs'], dtype=np.float32)

    # ArUco transformation to world
    with open('./aruco_final_summary.json', 'r') as f:
        aruco_dict_json = json.load(f)

    serial_to_view = {
        41182735: 'front',
        49429257: 'right',
        44377151: 'left',
        49045152: 'top'
    }
    view = serial_to_view.get(serial)
    ar_t, ar_r = None, None
    for entry in aruco_dict_json:
        if entry['view'] == view and entry['cam'] == 'leftcam':
            ar_t = np.array([entry['mean_x'], entry['mean_y'], entry['mean_z']], dtype=np.float32)
            ar_r = np.deg2rad([entry['rvec_x_deg'], entry['rvec_y_deg'], entry['rvec_z_deg']])
            break

    if ar_t is None or ar_r is None:
        print(f"[ERROR] ArUco pose not found for view '{view}'")
        return

    # ArUco detection setup
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    detector_params = aruco.DetectorParameters()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        frame = img.get_data()[:, :, :3].copy()
        frame_undistorted = cv2.undistort(frame, K, dist)

        # --- Forward Kinematics projection ---
        with joint_lock:
            joints = list(shared_joints)
        coords3d = forward_kinematics(joints)
        pts2d = project_to_image(coords3d, ar_r, ar_t, K, dist).astype(int)

        for i in range(len(pts2d) - 1):
            cv2.line(frame_undistorted, tuple(pts2d[i]), tuple(pts2d[i + 1]), (255, 0, 255), 2)
        for idx, pt in enumerate(pts2d):
            cv2.circle(frame_undistorted, tuple(pt), 5, (0, 255, 255), -1)
            cv2.putText(frame_undistorted, f"J{idx}", (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- ArUco Detection & Pose Estimation ---
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

        if ids is not None and len(ids) > 0:
            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])
                corner = corner.reshape((4, 2))

                # Subpixel refinement
                cv2.cornerSubPix(gray, corner, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

                # Pose estimation
                ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, K, dist)
                rvec, tvec = cv2.solvePnPRefineLM(marker_3d_edges, corner, K, dist, rvec, tvec)
                if not ret:
                    continue

                # Visualization
                for pt in corner:
                    cv2.circle(frame_undistorted, tuple(pt.astype(int)), 4, (255, 0, 0), -1)

                cv2.drawFrameAxes(frame_undistorted, K, dist, rvec, tvec, marker_size / 2)
                cv2.putText(frame_undistorted, f"ID: {marker_id}",
                            (int(corner[0][0]), int(corner[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame_undistorted, "ArUco Not Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Put frame into queue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame_undistorted)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == '__main__':
    robot = mdr.Robot()
    robot.Connect(address="192.168.0.100", enable_synchronous_mode=False)
    robot.ActivateAndHome()
    robot.WaitHomed()
    robot.SetJointVel(10)
    print("Robot activated, homed, and ready.")

    shared_joints = [0.0] * 7
    joint_lock = threading.Lock()

    t_joint = threading.Thread(target=update_joints, args=(robot, shared_joints, joint_lock), daemon=True)
    t_joint.start()

    t_motion = threading.Thread(target=robot_motion_loop, args=(robot,), daemon=True)
    t_motion.start()

    # ZED 카메라 구성: 시리얼 번호 → (보정파일 경로, 창 이름)
    cams = {
        41182735: ('./calib/front_41182735_leftcam_calib.json', 'Front View'),
        # 49429257: ('./calib/right_49429257_leftcam_calib.json', 'Right View'),
        # 44377151: ('./calib/left_44377151_leftcam_calib.json', 'Left View'),
        # 49045152: ('./calib/top_49045152_leftcam_calib.json', 'Top View'),
    }

    frame_queues = {}
    threads = []

    for serial, (calib_path, win_name) in cams.items():
        q = Queue(maxsize=1)
        frame_queues[win_name] = q
        t = threading.Thread(target=camera_thread, args=(serial, calib_path, shared_joints, joint_lock, q), daemon=True)
        threads.append(t)
        t.start()

    print("Press 'q' in any window to exit.")
    try:
        while True:
            for win_name, q in frame_queues.items():
                if not q.empty():
                    cv2.imshow(win_name, q.get())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received.")

    cv2.destroyAllWindows()
    robot.DeactivateRobot()
    robot.Disconnect()
    print("[INFO] Shutdown complete.")


