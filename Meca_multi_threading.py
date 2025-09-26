#!/usr/bin/env python3
# vla_robot_control.py

import logging
import time
import threading

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

# 모니터링 스레드 실행 여부를 제어하기 위한 플래그
monitoring_active = True

def get_vla_target_coordinate():
    """
    [VLA 연동 파트]
    VLA로부터 다음 목표 EE 좌표(x, y, z, alpha, beta, gamma)를 받아옵니다.
    지금은 사용자의 키보드 입력을 통해 좌표를 시뮬레이션합니다.
    """
    input_str = input("VLA 목표 좌표 입력 (예: 200, 0, 300, 0, 90, 0) 또는 'q'로 종료: ")
    if input_str.lower() == 'q':
        return None
    try:
        parts = [float(p.strip()) for p in input_str.split(',')]
        return parts
    except ValueError:
        print("잘못된 형식입니다. 숫자 6개를 쉼표로 구분하여 입력하세요.")
        return get_vla_target_coordinate()

def monitor_robot_state(robot: mdr.Robot):
    """
    [모니터링 스read]
    별도의 스레드에서 실행되며, 로봇의 현재 관절 각도를 주기적으로 읽어와 출력합니다.
    """
    logger = logging.getLogger("monitor_thread")
    logger.info("모니터링 스레드 시작.")
    while monitoring_active:
        try:
            # GetRt... API는 실시간 데이터를 빠르게 가져옵니다.
            joints = robot.GetRtTargetJointPos(synchronous_update=True)
            print(f"[MONITOR] Joints: {joints.data}") # .data로 실제 값에 접근
            # time.sleep(0.1) # 0.1초 간격으로 조회
        except mdr.MecademicException as e:
            # 메인 스레드에서 연결이 끊기면 예외 발생 가능
            logger.warning(f"모니터링 중 예외 발생: {e}")
            break
    logger.info("모니터링 스레드 종료.")
    return joints.data


def main():
    """
    [메인 스레드]
    로봇 연결 및 초기화를 수행하고, VLA로부터 좌표를 받아 이동 명령을 내립니다.
    """
    global monitoring_active
    tools.SetDefaultLogger(logging.INFO)
    logger = logging.getLogger("main_thread")

    with initializer.RobotWithTools() as robot:
        try:
            # 1. 로봇 연결 및 초기화
            robot.Connect(address="192.168.0.100")
            logger.info("로봇 연결됨.")

            initializer.reset_sim_mode(robot)
            initializer.reset_motion_queue(robot, activate_home=True)
            robot.WaitHomed()
            logger.info("로봇 활성화 및 호밍 완료.")

            # 2. 모니터링 스레드 시작
            monitor_thread = threading.Thread(target=monitor_robot_state, args=(robot,))
            monitor_thread.start()

            # 3. VLA 좌표 기반 이동 루프
            while True:
                target_pose = get_vla_target_coordinate()

                if target_pose is None:
                    logger.info("종료 명령을 받았습니다.")
                    break

                # MovePose API에 좌표 전달 (좌표계는 로봇 설정에 따름)
                logger.info(f"VLA 목표 {target_pose}로 이동합니다...")
                robot.MovePose(*target_pose)
                robot.WaitIdle() # 움직임이 끝날 때까지 대기
                logger.info("목표 도달 완료.")

        except mdr.MecademicException as e:
            logger.error(f"Mecademic API 오류 발생: {e}")
        except KeyboardInterrupt:
            logger.warning("사용자 중단.")
        finally:
            # 4. 종료 처리
            monitoring_active = False # 모니터링 스레드 종료 신호
            if 'monitor_thread' in locals() and monitor_thread.is_alive():
                monitor_thread.join() # 스레드가 완전히 끝날 때까지 대기

            if robot.IsConnected():
                if robot.GetStatusRobot().error_status:
                    robot.ResetError()
                robot.DeactivateRobot()
                robot.Disconnect()
                logger.info("로봇 비활성화 및 연결 해제 완료.")

if __name__ == "__main__":
    main()