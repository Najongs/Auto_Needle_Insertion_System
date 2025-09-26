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
import logging
from queue import Queue, Empty

# -------------------------------------------------------------------
# 로깅 및 전역 변수 설정
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
app_running = True

# -------------------------------------------------------------------
# 순기구학 / 투영 함수 (이전과 동일)
# -------------------------------------------------------------------
def get_dh_matrix(a, d, alpha, theta):
    alpha_rad, theta_rad = np.deg2rad(alpha), np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad) * np.cos(alpha_rad),  np.sin(theta_rad) * np.sin(alpha_rad), a * np.cos(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad) * np.cos(alpha_rad), -np.cos(theta_rad) * np.sin(alpha_rad), a * np.sin(theta_rad)],
        [0,                  np.sin(alpha_rad),                     np.cos(alpha_rad),                      d],
        [0,                  0,                                     0,                                      1]
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
    joint_coords = [np.array([0, 0, 0])]
    T = np.eye(4)
    rot_z_90 = R.from_euler('z', 90, degrees=True)
    T[:3, :3] = rot_z_90.as_matrix() # 베이스 프레임 보정
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
    img_pts, _ = cv2.projectPoints(joints_3d, rvec, tvec, K, dist)
    return img_pts.reshape(-1, 2).astype(int)

# -------------------------------------------------------------------
# 로봇 제어 클래스 (안정성 강화 버전, 수정 없음)
# -------------------------------------------------------------------
class RobotController:
    def __init__(self, ip="192.168.0.100"):
        self.robot = mdr.Robot()
        self.ip = ip
        self.current_joints = np.zeros(7) # FK 호환성을 위해 7로 유지
        self.lock = threading.Lock()
        
        self.robot.Connect(address=self.ip, enable_synchronous_mode=False)
        logging.info(f"로봇 {self.ip}에 연결되었습니다.")
        self.robot.ActivateAndHome()
        self.robot.WaitHomed()
        logging.info("로봇 활성화 및 호밍 완료.")
    
    @property
    def running(self):
        global app_running
        return app_running

    def start(self, poses_to_cycle, joint_speed):
        self.robot.SetJointVel(joint_speed)
        self.status_thread = threading.Thread(target=self.update_robot_status)
        self.motion_thread = threading.Thread(target=self.robot_motion_loop, args=(poses_to_cycle,))
        self.status_thread.start()
        self.motion_thread.start()

    def update_robot_status(self):
        while self.running:
            try:
                status = self.robot.GetStatusRobot()
                if status.error_status:
                    logging.error("!!!!!! 로봇 에러 감지! 자동 리셋을 시도합니다. !!!!!!")
                    last_msg = self.robot.GetNextMsg()
                    logging.error(f"  -> 에러 내용: {last_msg}")
                    self.robot.ResetError()
                    self.robot.ClearMotion()
                    time.sleep(1)

                joints = self.robot.GetJoints()
                joints.append(0.0)
                with self.lock:
                    self.current_joints[:] = joints
                time.sleep(0.05)
            except mdr.MecademicException as e:
                if self.running: logging.error(f"로봇 상태 업데이트 중 예외 발생: {e}")
                break
    
    def robot_motion_loop(self, poses):
        self.robot.SetCartAngVel(30)
        self.robot.SetCartLinVel(100)
        checkpoint = self.robot.SetCheckpoint(1)

        while self.running:
            for pose in poses:
                if not self.running: break
                try:
                    logging.info(f"다음 목표 {pose}로 이동 시작...")
                    self.robot.MoveJoints(*pose)
                    # checkpoint = self.robot.SetCheckpoint(1)
                    # checkpoint.wait(timeout=5)
                    logging.info(f"  -> 목표 도달 완료.")
                    # time.sleep(1)
                except mdr.MecademicException as e:
                    logging.warning(f"  -> 목표 {pose}로 이동 실패: {e}. 다음 목표로 넘어갑니다.")
                    self.robot.ClearMotion()
                    continue
    
    def get_current_joints(self):
        with self.lock:
            return self.current_joints.copy()

    def shutdown(self):
        if self.robot and self.robot.IsConnected():
            self.robot.DeactivateRobot()
            self.robot.Disconnect()
            logging.info("로봇 연결 해제 완료.")

# -------------------------------------------------------------------
# 메인 함수 (사용자 요청에 따라 수정됨)
# -------------------------------------------------------------------
def main():
    global app_running
    robot_controller = None

    # --- 1. 카메라 설정 ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution=sl.RESOLUTION.SVGA
    init_params.coordinate_units=sl.UNIT.METER
    init_params.depth_mode=sl.DEPTH_MODE.NEURAL
    init_params.camera_fps = 60
    init_params.set_from_serial_number(41182735)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise Exception("ZED 카메라 열기 실패")

    cam_info = zed.get_camera_information().camera_configuration
    cam_params = cam_info.calibration_parameters.left_cam
    K = np.array([[cam_params.fx, 0, cam_params.cx], [0, cam_params.fy, cam_params.cy], [0, 0, 1]])
    dist = cam_params.disto

    # --- 2. ArUco 설정 ---
    marker_size = 0.05
    marker_3d_edges = np.float32([[0,0,0], [0,marker_size,0], [marker_size,marker_size,0], [marker_size,0,0]]).reshape((4,1,3))
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector_params = cv2.aruco.DetectorParameters()

    threads = []
    try:
        # --- 3. 로봇 설정 및 시작 ---
        robot_poses = [
            (0, -60, 60, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (-5, -24, 37, 1, 44, 60),
        ]
        robot_controller = RobotController()
        robot_controller.start(poses_to_cycle=robot_poses, joint_speed=5)
        
        # 스레드 리스트에 추가 (안전한 종료를 위함)
        threads.append(robot_controller.status_thread)
        threads.append(robot_controller.motion_thread)

        # --- 4. 메인 루프 (시각화) ---
        image_zed = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        while app_running:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                frame = image_zed.get_data()[:, :, :3].copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
                
                if ids is not None and 3 in ids:
                    idx = np.where(ids == 3)[0][0]
                    ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corners[idx], K, dist)
                    if ret:
                        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_size / 2)
                        current_joints = robot_controller.get_current_joints()
                        coords3d = forward_kinematics(current_joints)
                        pts2d = project_to_image(coords3d, rvec, tvec + [[0],[0],[0.075]], K, dist)

                        if pts2d.size > 0:
                            for i in range(len(pts2d) - 1):
                                cv2.line(frame, tuple(pts2d[i]), tuple(pts2d[i+1]), (255, 0, 255), 2)
                            for pt in pts2d:
                                cv2.circle(frame, tuple(pt), 6, (255, 0, 255), -1)

                cv2.imshow("Live Robot Visualization", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    app_running = False
            else:
                time.sleep(0.001)
    
    except KeyboardInterrupt:
        logging.info("사용자 중단 (Ctrl+C). 프로그램을 종료합니다.")
        app_running = False
    finally:
        logging.info("종료 절차 시작...")
        for t in threads:
            if t.is_alive():
                t.join(timeout=1)
        if robot_controller:
            robot_controller.shutdown()
        zed.close()
        cv2.destroyAllWindows()
        logging.info("모든 리소스 정리 완료.")


if __name__ == "__main__":
    main()