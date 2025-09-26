import pyzed.sl as sl
import cv2
import threading
import time
import os
import json
import sys

# 저장 폴더
OUTPUT_DIR = "./dataset/ZED_Captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ZedCamera(threading.Thread):
    def __init__(self, serial_number, output_subdir, start_event, stop_event, duration=30):
        super().__init__()
        self.serial_number = serial_number
        self.output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.start_event = start_event
        self.duration = duration
        self.stop_event = stop_event
        self.ready = False


    def init_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200 # HD1200
        init_params.camera_fps = 30
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.ready = True
            print(f"Camera {self.serial_number} initialized")
        else:
            print(f"Failed to open camera {self.serial_number}")
            sys.exit(1)

    def run(self):
        if not self.ready:
            print(f"Camera {self.serial_number} not ready. Skipping capture.")
            return

        left_image = sl.Mat()
        right_image = sl.Mat()

        self.start_event.wait()
        print(f"Camera {self.serial_number} started capturing")
        start_time = time.time()

        while time.time() - start_time < self.duration or not self.stop_event.is_set():
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                timestamp_left = time.time()
                timestamp_left_str = f"{timestamp_left:.3f}"
                
                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                timestamp_right = time.time()
                timestamp_right_str = f"{timestamp_right:.3f}"
                
                left_data = left_image.get_data()
                right_data = right_image.get_data()
                
                left_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_left_{timestamp_left_str}.jpg")
                right_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_right_{timestamp_right_str}.jpg")

                success_left = cv2.imwrite(left_path, left_data[:, :, :3])
                success_right = cv2.imwrite(right_path, right_data[:, :, :3])

                if not (success_left and success_right):
                    print(f"Camera {self.serial_number} - Failed to save images at {timestamp_right_str:.3f}")

        self.zed.close()
        print(f"Camera {self.serial_number} stopped")

import logging
import pathlib
from datetime import datetime

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools


class RobotManager:
    def __init__(self, address="192.168.0.100"):
        self.address = address
        self.robot = None
        self.logger = logging.getLogger(__name__)
        self._log_cm = None  # FileLogger 컨텍스트 핸들
        self._log_active = False

    def __enter__(self):
        tools.SetDefaultLogger(logging.INFO, f'{pathlib.Path(__file__).stem}.log')
        self.robot = initializer.RobotWithTools()
        self.robot.__enter__()
        self.robot.Connect(address=self.address, disconnect_on_exception=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._log_active:
            try:
                self._log_cm.__exit__(None, None, None)
                self.logger.info('FileLogger stopped (in __exit__).')
            except Exception as e:
                self.logger.warning(f'logger close failed in __exit__: {e}')
            finally:
                self._log_active = False
                self._log_cm = None

        # 2) 로봇 정리
        if self.robot and self.robot.IsConnected():
            try:
                if self.robot.GetStatusRobot().error_status:
                    self.logger.info('Robot has encountered an error, attempting to clear...')
                    self.robot.ResetError()
                    self.robot.ResumeMotion()
            except Exception as e:
                self.logger.warning(f'Error check/clear failed: {e}')
            try:
                self.robot.DeactivateRobot()
                self.logger.info('Robot is deactivated.')
            except Exception as e:
                self.logger.warning(f'Deactivate failed: {e}')
        if self.robot:
            self.robot.__exit__(exc_type, exc_value, traceback)


    def setup(self):
        """로봇 초기화 & homing"""
        self.logger.info('Activating and homing robot...')
        initializer.reset_sim_mode(self.robot)
        initializer.reset_motion_queue(self.robot, activate_home=True)
        initializer.reset_vacuum_module(self.robot)
        self.robot.WaitHomed()
        self.logger.info('Robot is homed and ready.')
        self.robot.SetCartLinVel(100)
        self.robot.SetJointVel(1)
        self.robot.SetBlending(50)
        self.robot.MoveJoints(*([0] * self.robot.GetRobotInfo().num_joints))
        self.robot.WaitIdle(30)

    def move_points(self, points):
        """포인트 리스트대로 이동"""
        if tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model):
            self.robot.SetConf(1, 1, 1)
            for idx, (x, y, z, a, b, r) in enumerate(points):
                self.logger.info(f'Moving to point {idx+1}: ({x}, {y}, {z})')
                self.robot.MovePose(x, y, z, a, b, r)
                self.robot.WaitIdle(60)
        else:
            raise mdr.MecademicException(
                f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}'
            )
        self.logger.info('Finished executing move_points.')
    
    def start_logging(self, period=0.01, filename_prefix="trajectory",
                      base_fields=None):
        if self._log_active:
            self.logger.info('Logging already active; ignoring start_logging().')
            return

        if base_fields is None:
            base_fields = ["TargetJointPos", "JointPos"]

        ts_candidates = ("TimeStamp", "Timestamp", "Time")
        last_err = None
        ts_used = None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{filename_prefix}_{stamp}"

        for ts in ts_candidates:
            try:
                fields = [ts] + base_fields
                cm = self.robot.FileLogger(period, fields=fields, file_name=file_name)
                cm.__enter__()  # 수동 진입: 컨텍스트 유지(로봇 동작 동안 계속 기록)
                self._log_cm = cm
                self._log_active = True
                ts_used = ts
                self.logger.info(f"FileLogger started: {file_name}.csv (fields={fields})")
                break
            except Exception as e:
                last_err = e
                continue

        if not self._log_active:
            self.logger.error(
                f"Failed to start FileLogger with timestamp fields. "
                f"Last error: {last_err}"
            )
            raise

        return ts_used, file_name
    
    def stop_logging(self):
        if not self._log_active:
            self.logger.info('Logging is not active; ignoring stop_logging().')
            return
        try:
            self._log_cm.__exit__(None, None, None)  # EndLogging
            self.logger.info('FileLogger stopped.')
        except Exception as e:
            self.logger.warning(f'FileLogger stop failed: {e}')
        finally:
            self._log_active = False
            self._log_cm = None



def main():
    start_event = threading.Event()
    stop_event = threading.Event()
    duration = 30  # 30초 동안 촬영

    cameras = [
        ZedCamera(serial_number=41182735, output_subdir="view1", start_event=start_event, stop_event=stop_event, duration=duration),
        ZedCamera(serial_number=49429257, output_subdir="view2", start_event=start_event, stop_event=stop_event, duration=duration),
        ZedCamera(serial_number=44377151, output_subdir="view3", start_event=start_event, stop_event=stop_event, duration=duration),
        ZedCamera(serial_number=49045152, output_subdir="view4", start_event=start_event, stop_event=stop_event, duration=duration),
    ]

    for cam in cameras:
        cam.init_camera()

    while not all(cam.ready for cam in cameras):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)

    for cam in cameras:
        cam.start()

    print("Starting data capture in 3 seconds...")
    time.sleep(2)
    start_event.set()

    try:
        with RobotManager() as manager:
            manager.setup()
            manager.start_logging(period=0.01, filename_prefix="trajectory")
            manager.move_points([
                (245.330857, 71.314158, 70.579333, -174.999591, 14.169026, 161.421036)
            ])
            stop_event.set()
            manager.stop_logging()
    finally:
        for cam in cameras:
            cam.join()
        print("Data collection finished.")

if __name__ == "__main__":
    main()

# white block:
#     joint angles: 15.991698, 65.637785, -23.417432, 1.460125, 32.800052, -3.088784\
#     EE pose: 245.330857, 71.314158, 70.579333, -174.999591, 14.169026, 161.421036


# 로봇 시간 후처리 
# import pandas as pd
# from datetime import datetime, timedelta
# start_time = datetime.now()
# df = pd.read_csv("trajectory_20250926_153000.csv")
# df["WallClock"] = [start_time + timedelta(seconds=t) for t in df["TimeStamp"]]
# print(df.head())