# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.mimic_leader.mimic_leader import MimicLeader
from lerobot.teleoperators.mimic_leader.config_mimic_leader import MimicLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Initialize the robot and teleoperator config
    follower_config = SO100FollowerConfig(
        port="/dev/ttyACM0", id="so_101_follower_arm", use_degrees=True
    )
    # Configure and initialize follower and mimic leader (bimanual)
    # NOTE: Provide correct ports for left and right arms here. If you only have one
    # teleop device, you can set both ports to the same device, otherwise supply
    # distinct device paths.
    leader_config = MimicLeaderConfig(left_arm_port="/dev/tty.usbmodem5A460819811", right_arm_port="/dev/tty.usbmodem5A460819812", id="my_awesome_mimic_leader")

    # Initialize the robot and teleoperator
    follower = SO100Follower(follower_config)
    leader = MimicLeader(leader_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    follower_kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # For MimicLeader we use the right arm to drive the follower by default.
    # Use leader.right_arm.bus.motors to build the kinematics joint list.
    leader_kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.right_arm.bus.motors.keys()),
    )

    # Build pipeline to convert teleop joints to EE action
    leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # build pipeline to convert EE action to robot joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    follower.connect()
    leader.connect()

    # Init rerun viewer
    init_rerun(session_name="so100_so100_EE_teleop")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = follower.get_observation()

        # Get teleop observation (MimicLeader returns combined left/right/base keys)
        leader_joints_obs = leader.get_action()

        # Extract right-arm joint commands (keys are prefixed with 'right_') and strip prefix
        if isinstance(leader_joints_obs, dict):
            right_action = {k.removeprefix("right_"): v for k, v in leader_joints_obs.items() if k.startswith("right_")}
            # Fallback: if no right_ keys exist, try using full observation as-is
            leader_input_for_fk = right_action if right_action else leader_joints_obs
        else:
            leader_input_for_fk = leader_joints_obs

        # teleop joints -> teleop EE action (use right arm of the mimic leader)
        leader_ee_act = leader_to_ee(leader_input_for_fk)

        # teleop EE -> robot joints
        follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))

        # Send action to robot
        _ = follower.send_action(follower_joints_act)

        # Visualize
        log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
