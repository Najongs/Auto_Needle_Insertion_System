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
            except Exception as e:
                self.logger.warning(f'logger close failed: {e}')
            finally:
                self._log_active = False
                self._log_cm = None

        if self.robot and self.robot.IsConnected():
            if self.robot.GetStatusRobot().error_status:
                self.logger.info('Robot has encountered an error, attempting to clear...')
                self.robot.ResetError()
                self.robot.ResumeMotion()
            self.robot.DeactivateRobot()
            self.logger.info('Robot is deactivated.')
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
        self.robot.SetJointVel(5)
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
                self.robot.WaitIdle(30)
        else:
            raise mdr.MecademicException(
                f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}'
            )
        self.logger.info('Finished executing move_points.')
    
    def start_logging(self, period=0.001, filename_prefix="trajectory",
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
            self._log_cm.__exit__(None, None, None)
            self.logger.info('FileLogger stopped.')
        finally:
            self._log_active = False
            self._log_cm = None

if __name__ == "__main__":
    with RobotManager() as manager:
        manager.setup()
        manager.start_logging(period=0.001, filename_prefix="trajectory")
        manager.move_points([[245.330857, 71.314158, 70.579333, -174.999591, 14.169026, 161.421036]])
        manager.stop_logging()
        manager.__exit__

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
