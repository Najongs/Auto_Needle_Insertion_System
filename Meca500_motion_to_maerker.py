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
        # {'alpha': 0,   'a': 0,     'd': 0.130, 'theta_offset': 0}, # End-effector (Tool)
    ]
    joint_coords = [np.array([0, 0, 0])]
    T = np.eye(4)
    rot_x_180 = R.from_euler('x', 180, degrees=True)
    rot_z_90 = R.from_euler('z', 90, degrees=True)
    T[:3, :3] = (rot_z_90 * rot_x_180).as_matrix()
    base_point = np.array([[0], [0], [0], [1]])
    for i in range(6):
        p = dh_params[i]
        theta = joint_angles[i] + p['theta_offset']
        T_i = get_dh_matrix(p['a'], p['d'], p['alpha'], theta)
        T = T @ T_i
        joint_pos = T @ base_point
        joint_coords.append(joint_pos[:3, 0])
    return np.array(joint_coords, dtype=np.float32)

def project_to_image(joints_3d, rvec, tvec, K, dist):
    """Project 3D joints into 2D image using extrinsics and intrinsics."""
    # Ensure joints_3d is a 2D array
    if joints_3d.ndim == 1:
        joints_3d = joints_3d.reshape(1, -1)
    R_mat, _ = cv2.Rodrigues(rvec)
    pts_cam = (R_mat @ joints_3d.T).T + tvec.reshape(1, 3)
    img_pts, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    return img_pts.reshape(-1, 2)

# -------------------------------------------------------------------
# Robot & Camera Threads
# -------------------------------------------------------------------
def update_joints(robot, shared_joints, joint_lock):
    """Continuously update joint positions from robot."""
    while True:
        try:
            joints = robot.GetJoints()
            # joints.append(0.0) # 7th dummy joint for FK compatibility
            with joint_lock:
                shared_joints[:] = joints
            time.sleep(0.05)
        except mdr.MecademicException:
            print("[INFO] Joint update thread stopped.")
            break

def robot_motion_loop(robot, motion_params, shared_data, locks):
    """Move robot to a pose relative to the detected ArUco marker using proportional delta movements."""
    print("[INFO] Robot motion loop started. Waiting for ArUco marker detection...")

    # --- NEW: 수동 오프셋 값 (mm 단위) ---
    MANUAL_OFFSET_MM = np.array([-200.0, -300.0, -500.0])

    while True:
        with locks['aruco']:
            if not shared_data['aruco']['new_pose_available']:
                time.sleep(0.1)
                continue
            
            tvec_cam_aruco = shared_data['aruco']['tvec'].copy()
            rvec_cam_aruco = shared_data['aruco']['rvec'].copy()
            shared_data['aruco']['new_pose_available'] = False

        # 1. 절대 목표 위치 계산
        R_cam_aruco, _ = cv2.Rodrigues(rvec_cam_aruco)
        T_cam_aruco = np.eye(4)
        T_cam_aruco[:3, :3] = R_cam_aruco
        T_cam_aruco[:3, 3] = tvec_cam_aruco.flatten()
        T_base_aruco = motion_params['T_base_camera'] @ T_cam_aruco

        T_base_aruco[:2, 2] *= 1 # Y축 반전

        target_pos_m = T_base_aruco[:3, 3]
        target_rot_matrix = T_base_aruco[:3, :3]
        
        target_pos_m[2] += motion_params['Z_OFFSET']
        
        target_pos_mm = target_pos_m * 1000

        # --- NEW: 계산된 좌표에 수동 오프셋 적용 ---
        target_pos_mm += MANUAL_OFFSET_MM
        
        r = R.from_matrix(target_rot_matrix)
        target_euler_deg = np.array(r.as_euler('xyz', degrees=True))
        
        # 시각화를 위해 m 단위로 다시 변환하여 공유
        with locks['aruco']:
            shared_data['aruco']['target_pos_m'] = target_pos_mm / 1000.0

        try:
            # (이후 J6 계산, Delta 계산, 이동 로직은 동일)
            with locks['joint']:
                current_angles_deg = list(shared_data['joints'])
            all_joint_coords_m = forward_kinematics(current_angles_deg)
            j6_pos_mm = all_joint_coords_m[5] * 1000
            print("-" * 30)
            print(f"Target (Offset) (mm): x={target_pos_mm[0]:.2f}, y={target_pos_mm[1]:.2f}, z={target_pos_mm[2]:.2f}")
            print(f"Joint 6 Pose    (mm): x={j6_pos_mm[0]:.2f}, y={j6_pos_mm[1]:.2f}, z={j6_pos_mm[2]:.2f}")

            current_pose_list = robot.GetPose()
            current_pos_mm = np.array(current_pose_list[:3])
            
            delta_pos = target_pos_mm - current_pos_mm
            
            distance_to_target = np.linalg.norm(delta_pos)
            if distance_to_target < motion_params['MOVE_THRESHOLD_MM']:
                time.sleep(0.1)
                continue
            
            # 회전(delta_euler)은 잠시 0으로 고정하여 위치만 제어
            delta_euler = np.array([0, 0, 0]) 
            full_delta_pose = np.concatenate((delta_pos, delta_euler))
            move_delta = full_delta_pose * motion_params['P_GAIN']

            print(f"[MOTION] Moving proportionally by delta: dx={move_delta[0]:.2f}, dy={move_delta[1]:.2f}, dz={move_delta[2]:.2f}...")
            robot.MoveLinRelWrf(*move_delta) 
            robot.WaitIdle()

        except mdr.MecademicException as e:
            print(f"[ERROR] Robot motion failed: {e}")
            break
            
        time.sleep(0.1)
    print("[INFO] Robot motion thread stopped.")

def camera_thread(cam_info, shared_data, locks, frame_queue):
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(cam_info['serial'])
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open camera {cam_info['serial']}")
        return
    
    runtime = sl.RuntimeParameters()
    K = cam_info['K']
    dist = cam_info['dist']
    ar_t = cam_info['ar_t']
    ar_r = cam_info['ar_r']
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    detector_params = aruco.DetectorParameters()
    marker_size = 0.05
    marker_3d_edges = np.array([
        [-marker_size/2, marker_size/2, 0], [marker_size/2, marker_size/2, 0],
        [marker_size/2, -marker_size/2, 0], [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue

        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        frame = img.get_data()[:, :, :3].copy()
        frame_undistorted = cv2.undistort(frame, K, dist)

        # --- IMPROVEMENT: Consistent lock order (aruco -> joint) to prevent deadlock ---
        with locks['aruco']:
            target_pos_3d_m = shared_data['aruco'].get('target_pos_m')
            with locks['joint']:
                joints = list(shared_data['joints'])
        
        # Draw current robot pose
        coords3d = forward_kinematics(joints)
        pts2d = project_to_image(coords3d, ar_r, ar_t, K, dist).astype(int)
        for i in range(len(pts2d) - 1):
            cv2.line(frame_undistorted, tuple(pts2d[i]), tuple(pts2d[i + 1]), (255, 0, 255), 2)
        for idx, pt in enumerate(pts2d):
            if idx == len(pts2d) - 1:
                cv2.circle(frame_undistorted, tuple(pt), 6, (0, 0, 255), -1)
                cv2.putText(frame_undistorted, "End-Effector", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.circle(frame_undistorted, tuple(pt), 5, (0, 255, 255), -1)
        
        # Detect ArUco marker
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        if ids is not None:
            ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corners[0], K, dist)
            if ret:
                with locks['aruco']:
                    shared_data['aruco']['tvec'] = tvec
                    shared_data['aruco']['rvec'] = rvec
                    shared_data['aruco']['new_pose_available'] = True
                cv2.drawFrameAxes(frame_undistorted, K, dist, rvec, tvec, marker_size / 2)

        # Draw target pose
        if target_pos_3d_m is not None:
            target_pt_2d = project_to_image(target_pos_3d_m, ar_r, ar_t, K, dist)
            if target_pt_2d is not None and len(target_pt_2d) > 0:
                pt = tuple(target_pt_2d[0].astype(int))
                cv2.circle(frame_undistorted, pt, 10, (0, 255, 0), 2)
                cv2.line(frame_undistorted, (pt[0]-10, pt[1]), (pt[0]+10, pt[1]), (0, 255, 0), 2)
                cv2.line(frame_undistorted, (pt[0], pt[1]-10), (pt[0], pt[1]+10), (0, 255, 0), 2)
                cv2.putText(frame_undistorted, "Target", (pt[0] + 15, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not frame_queue.full():
            frame_queue.put(frame_undistorted)
    print(f"[INFO] Camera thread {cam_info['serial']} stopped.")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # --- IMPROVEMENT: Centralized configuration parameters ---
    # Robot and Camera settings
    ROBOT_IP = "192.168.0.100"
    CAMERA_CONFIG = {
        41182735: {'name': 'Front View', 'view': 'front'},
        # 49429257: {'name': 'Right View', 'view': 'right'}, # Add other cameras here if needed
    }
    
    # File paths
    CALIB_FILE_PATH = './calib'
    ARUCO_SUMMARY_FILE = './aruco_final_summary.json'
    
    # Motion control parameters
    MOTION_PARAMS = {
        'Z_OFFSET': -0.2,              # meters
        'MOVE_THRESHOLD_MM': 1.0,      # millimeters
        'P_GAIN': 0.01,                # Proportional gain (0.0 to 1.0)
        'JOINT_VEL_PERCENT': 0.5,        # Joint velocity percentage
    }
    # --- END IMPROVEMENT ---

    # Initialize Robot
    robot = mdr.Robot()
    try:
        robot.Connect(address=ROBOT_IP, enable_synchronous_mode=False)
        robot.ActivateAndHome()
        robot.WaitHomed()
        robot.SetJointVel(MOTION_PARAMS['JOINT_VEL_PERCENT'])
        print("Robot activated, homed, and ready.")
    except mdr.MecademicException as e:
        print(f"[FATAL] Robot connection failed: {e}")
        sys.exit(1)

    # Shared data and locks
    shared_data = {
        'joints': [0.0] * 6,
        'aruco': {'tvec': None, 'rvec': None, 'new_pose_available': False}
    }
    locks = {
        'joint': threading.Lock(),
        'aruco': threading.Lock()
    }

    # Load camera calibration and transformation data
    all_cam_info = {}
    with open(ARUCO_SUMMARY_FILE, 'r') as f:
        aruco_summary = json.load(f)
        
    for serial, config in CAMERA_CONFIG.items():
        calib_file = os.path.join(CALIB_FILE_PATH, f"{config['view']}_{serial}_leftcam_calib.json")
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        cam_info = {'serial': serial, 'name': config['name']}
        cam_info['K'] = np.array(calib_data['camera_matrix'], dtype=np.float32)
        cam_info['dist'] = np.array(calib_data['distortion_coeffs'], dtype=np.float32)

        for entry in aruco_summary:
            if entry['view'] == config['view'] and entry['cam'] == 'leftcam':
                cam_t = np.array([entry['mean_x'], entry['mean_y']-0.01, entry['mean_z']], dtype=np.float32)
                cam_r_rad = np.deg2rad([entry['rvec_x_deg'], entry['rvec_y_deg'], entry['rvec_z_deg']])
                cam_info['ar_t'] = cam_t
                cam_info['ar_r'] = cam_r_rad
                
                R_base_cam, _ = cv2.Rodrigues(cam_r_rad)
                T_base_camera = np.eye(4)
                T_base_camera[:3, :3] = R_base_cam
                T_base_camera[:3, 3] = cam_t
                MOTION_PARAMS['T_base_camera'] = T_base_camera # Add to motion params
                break
        all_cam_info[serial] = cam_info

    # Start threads
    threads = []
    frame_queues = {}
    
    t_joint = threading.Thread(target=update_joints, args=(robot, shared_data['joints'], locks['joint']), daemon=True)
    threads.append(t_joint)
    
    t_motion = threading.Thread(target=robot_motion_loop, args=(robot, MOTION_PARAMS, shared_data, locks), daemon=True)
    threads.append(t_motion)

    for serial, cam_info in all_cam_info.items():
        q = Queue(maxsize=2)
        frame_queues[cam_info['name']] = q
        t_cam = threading.Thread(target=camera_thread, args=(cam_info, shared_data, locks, q), daemon=True)
        threads.append(t_cam)

    for t in threads:
        t.start()

    # Main display loop
    print("Press 'q' in any window to exit.")
    try:
        while True:
            for win_name, q in frame_queues.items():
                if not q.empty():
                    cv2.imshow(win_name, q.get())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Check if motion thread is still alive
            if not t_motion.is_alive():
                print("[INFO] Motion thread has terminated. Exiting main loop.")
                break
    
    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received.")
    finally:
        print("[INFO] Shutting down...")
        cv2.destroyAllWindows()
        try:
            if robot.IsConnected():
                robot.DeactivateRobot()
                robot.Disconnect()
        except mdr.MecademicException as e:
            print(f"[ERROR] Error during robot shutdown: {e}")
        print("[INFO] Shutdown complete.")