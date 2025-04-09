from yumi_realtime.j_angle_controller import YuMiJointAngleROSInterface
from loguru import logger
import numpy as onp
import tyro
import rospy
from typing import Literal
from yumi_realtime.policy_controller.utils.utils import *
# from dp_gs.dataset.utils import *
from openpi.shared.eval_wrapper import OpenPIWrapper
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque
from scipy.spatial.transform import Rotation
import PIL.Image as PILimage
import torch
import time
import copy
from line_profiler import LineProfiler
from yumi_realtime.data_logging.data_collector import DataCollector
from std_srvs.srv import Empty, EmptyResponse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class YuMiPI0PolicyController(YuMiJointAngleROSInterface):
    """YuMi controller for pi0 policy control."""
    
    def __init__(self, ckpt_path: str = None, ckpt_id: int = 0, text_prompt: str = None, collect_data: bool = False, debug_mode = False, *args, **kwargs):
        super().__init__(slider_control=False, *args, **kwargs)

        assert ckpt_path is not None, "PI0 Policy checkpoint path must be provided."
        
        # Setup Diffusion Policy module and weights
        self.model = OpenPIWrapper(model_ckpt_folder=ckpt_path, ckpt_id=ckpt_id, text_prompt=text_prompt)
        self.collect_data = collect_data

        # ROS Camera Observation Subscriber
        self.bridge = CvBridge()
        self.height = None
        self.width = None
        
        self.proprio_buffer = deque([],maxlen=1)

        self.observation_buffers = {}
        self.camera_buffers = {}
        self.camera_topics = [topic[0] for topic in rospy.get_published_topics() if 'sensor_msgs/Image' in topic[1]]
        
        # Initialize a deque for each camera
        self.max_buffer_size = 5  # For storing recent messages to sync from
        self.main_camera = "camera_1"
        for idx, topic in enumerate(self.camera_topics):
            camera_name = f"camera_{topic.split('camera_')[1][0]}"
            if camera_name == self.main_camera:
                # Main camera's synchronized buffer
                self.observation_buffers[camera_name] = deque([], maxlen=1)
            else:
                # Other cameras' message buffers for synchronization
                self.camera_buffers[camera_name] = deque([], maxlen=self.max_buffer_size)
        main_topic = [topic for topic in self.camera_topics if self.main_camera in topic]
        other_topics = [topic for topic in self.camera_topics if self.main_camera not in topic]
        assert len(main_topic) == 1, f"There appears to be duplicately named ros topics containing main_camera keyword \"{self.main_camera}\""
        # Subscribe main camera separately since it drives synchronization
        rospy.Subscriber(main_topic[0], Image, self.main_camera_callback)
        
        # Subscribe other cameras
        for topic in other_topics:
            camera_name = f"camera_{topic.split('camera_')[1][0]}"
            rospy.Subscriber(topic, Image, self.camera_callback, callback_args=(camera_name,))
        
        self.cur_proprio = None
        
        # self.cartesian_pose_L = None
        # self.cartesian_pose_R = None
        
        # Control mode
        self.control_mode = 'receding_horizon_control'
        # self.control_mode = 'temporal_ensemble'
        
        self.skip_actions = 0
        if self.control_mode == 'receding_horizon_control':
            # self.max_len = self.model.model.action_horizon//2 #TODO: make action horizon a param for openpi wrapper
            self.max_len = 10
            self.action_queue = deque([],maxlen=self.max_len)
        elif self.control_mode == 'temporal_ensemble':
            self.action_queue = deque([],maxlen=10 - self.skip_actions) #TODO: make action horizon a param for openpi wrapper
        self.prev_action = deque([],maxlen=1)
        
        # self._setup_scene()
        
        logger.info("PI0 Policy controller initialized")

        if self.collect_data:
            self._setup_collectors()
            self.add_gui_data_collection_controls()

        self.gripper_thres = 0.01

        self.viser_img_handles = {}

        self.debug_mode = debug_mode
        if self.debug_mode:
            with self.server.gui.add_folder("Observation"):
                for camera_name in self.camera_topics:
                    self.viser_img_handles[camera_name] = self.server.gui.add_image(
                        image = onp.zeros((480, 848, 3)),
                        label = camera_name
                    )

        with self.server.gui.add_folder("State"):
            self.right_gripper_signal = self.server.gui.add_number("Right gripper pred.: ", 0.0, disabled=True)
            self.left_gripper_signal = self.server.gui.add_number("Left gripper pred.: ", 0.0, disabled=True)
        self.breakpoint_btn = self.server.gui.add_button("Breakpoint at Next Inference")
        
        self.breakpt_next_inference = False

        @self.breakpoint_btn.on_click
        def _(_) -> None:
            self.breakpt_next_inference = True

    def camera_callback(self, image_msg: Image, camera_name: str):
        """Store messages from non-main cameras for synchronization"""
        camera_name = camera_name[0]
        if camera_name not in self.camera_buffers:
            return
        
        self.camera_buffers[camera_name].append(image_msg)
    
    def main_camera_callback(self, image_msg: Image):
        """Synchronize all buffers based on main camera's frequency"""
        if self.height is None and self.width is None:
            self.height = image_msg.height
            self.width = image_msg.width
            logger.info(f"First image received from main camera; Observation dim: {self.height}x{self.width}x3")
        
        
        
        # if self.cartesian_pose_L is None or self.cartesian_pose_R is None:
        #     logger.info("fail to read the cartesian pose")
        #     return
        

        target_time = image_msg.header.stamp.to_nsec()
        
        # Process main camera
        onp_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8') #.astype("float32") / 255.0
        new_obs = onp.transpose(onp_img, (2, 0, 1))  # C, H, W
        
        # Get synchronized observations from other cameras
        all_cameras_ready = True
        synced_obs = {self.main_camera: new_obs}
        
        for camera_name, buffer in self.camera_buffers.items():
            if not buffer:  # Skip if buffer is empty
                all_cameras_ready = False
                break
                
            # Find closest message in time
            closest_msg = min(
                buffer,
                key=lambda msg: abs(msg.header.stamp.to_nsec() - target_time)
            )
            onp_img = self.bridge.imgmsg_to_cv2(closest_msg, desired_encoding='rgb8') #.astype("float32") / 255.0
            synced_obs[camera_name] = onp.transpose(onp_img, (2, 0, 1))
        
        if not all_cameras_ready:
            return
        
        # Update proprioception
        self.update_curr_proprio()
        
        # Update all image buffers
        for camera_name, obs in synced_obs.items():
            if camera_name == self.main_camera:
                self.observation_buffers[camera_name].append(obs)
            else:
                if camera_name not in self.observation_buffers:
                    self.observation_buffers[camera_name] = deque([obs] * (1), 
                                                        maxlen=1)
                self.observation_buffers[camera_name].append(obs)

    def run(self):
        """Diffusion Policy controller loop."""
        rate = rospy.Rate(150) # 150Hz control loop
        rate = None          
        self.home()
        i = 0
        start = time.time()
        while ((self.height is None or self.width is None) 
            #    or (self.cartesian_pose_L is None or self.cartesian_pose_R is None) 
               or time.time() - start < 7
               or len(self.proprio_buffer) != 1):
            self.home()
            if rate is not None:
                rate.sleep()
            # self.solve_ik()
            self.update_visualization()
            
            super().publish_joint_commands()
            if i % 400 == 0:
                self.call_gripper(side = 'left', gripper_state = False, enable = True)
                rospy.sleep(0.2)
                self.call_gripper(side = 'right', gripper_state = False, enable = True)
                logger.info("Waiting for camera topic or robot pose data...")
                i = 0
            i += 1
        
        rospy.sleep(1.5)
        if not rospy.is_shutdown():
            print("Press C to continue once finished homing")
            import pdb; pdb.set_trace()
            if self.collect_data:
                self.start_record()
                self.start_record_button.disabled = True
                self.save_success_button.disabled = False
                self.save_failure_button.disabled = False

        step = 0
        self.last_action = None
        while not rospy.is_shutdown():
            assert len(self.proprio_buffer) == 1
                        
            start = time.time()
            step += 1

            if self.last_action is not None: # TODO: fix blocking ctl for joint angle control
                # check gripper state of last action 
                target_left_gripper = self.last_action[-2] < self.gripper_thres
                target_right_gripper = self.last_action[-1] < self.gripper_thres

                current_left_gripper = input["proprio"][0, -1, -2] < self.gripper_thres
                current_right_gripper = input["proprio"][0, -1, -1] < self.gripper_thres
                
                print("current_left_gripper: ", current_left_gripper)

                # delta pose 
                target_proprio_left = self.last_action[:-2]
                current_proprio_left = input["proprio"][0, -1, :-2]
            
                # calculate lag 
                lag = onp.linalg.norm(target_proprio_left - current_proprio_left)
                print("lag: ", lag)

                # if they are in disgreemnt, gripper control with last action 
                # if target_left_gripper != current_left_gripper or target_right_gripper != current_right_gripper:
                if target_left_gripper != current_left_gripper or target_right_gripper != current_right_gripper or lag > 0.020:
                    print("blocking with last action")
                    self._yumi_control(self.last_action, rate)
                    self.last_action = None
                    continue

            # receding horizon control
            if self.control_mode == 'receding_horizon_control':
                if len(self.action_queue) > 0:
                    # self._update_action_queue_viz()
                    action = self.action_queue.popleft()
                    self._yumi_control(action, rate)
                    continue
            # end of receding horizon control

            all_camera_obs = []
            for camera_name in self.observation_buffers.keys():
                assert len(self.observation_buffers[camera_name]) == 1
                cam_obs = onp.array(self.observation_buffers[camera_name])
                all_camera_obs.append(cam_obs)
            
            # Stack along the N_C dimension
            stacked_obs = onp.stack(all_camera_obs, axis=1) # [T, N_C, C, H, W]
            if self.debug_mode:
                _img_0 = (stacked_obs[-1][0].transpose([1,2,0]))
                _img_1 = (stacked_obs[-1][1].transpose([1,2,0]))
                list(self.viser_img_handles.items())[0][1].image = _img_0
                list(self.viser_img_handles.items())[1][1].image = _img_1
                        
            stacked_obs = onp.transpose(onp.expand_dims(stacked_obs, axis=0), (0, 1, 2, 4, 5, 3)) # [T, N_C, C, H, W] -> [B, T, N_C, H, W, C]
            
            proprio_array = onp.expand_dims(onp.asarray(self.proprio_buffer), axis=0) # [T, 16] -> [B, T, 16]
            
            # Create input dict referencing pre-allocated tensors
            input = {
                "observation": stacked_obs,
                "proprio": proprio_array
            }

            inference_start = time.time()

            action_prediction = self.model(input) # PI0 inference step
            
            print("\nprediction called\n")
            print("Inference time: ", time.time() - inference_start)
            
            if self.breakpt_next_inference:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.plot_predictions(input["proprio"], action_prediction, timestamp)
                obs_copy = copy.deepcopy(stacked_obs)
                self.plot_stacked_obs(copy.deepcopy(obs_copy), timestamp)
                import pdb; pdb.set_trace()

            self.action_prediction = action_prediction

            # # temporal emsemble start
            if self.control_mode == 'temporal_ensemble':
                new_actions = deque(action_prediction[self.skip_actions:len(action_prediction)])
                self.action_queue.append(new_actions)
                actions_current_timestep = onp.empty((len(self.action_queue), action_prediction.shape[1]))
                
                k = 0.01

                for i, q in enumerate(self.action_queue):
                    actions_current_timestep[i] = q.popleft()

                exp_weights = onp.exp(k * onp.arange(actions_current_timestep.shape[0]))
                exp_weights = exp_weights / exp_weights.sum()

                action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
                self.temporal_ensemble_action = action
                
            # receding horizon # check the receding horizon block as well
            if self.control_mode == 'receding_horizon_control':
                if len(self.action_queue) == self.max_len: # If at max, discard a few due to latency
                    for i in range(self.skip_actions):
                        self.action_queue.popleft()
                    for i in range(5):
                        self.action_queue.pop()
                        
                if len(self.action_queue) == 0: 
                    self.action_queue = deque([a for a in action_prediction[:self.max_len]])
                action = self.action_queue.popleft()
            
            # update yumi action 
            self._yumi_control(action, rate)
            # self._update_action_queue_viz()
    
    def _yumi_control(self, action, rate = None):
        # YuMi action update
        # TODO: Rewrite this to be (16,) joint positions in format 
        ######################################################################
        print("action update called")
        self.last_action = action
        
        # Map the action to the correct joint order for the robot
        mapped_action = self._map_action_to_robot_joints(action)
        
        # Extract gripper commands
        l_gripper_cmd = mapped_action[15]  # Left gripper is at index 14
        r_gripper_cmd = mapped_action[14]  # Right gripper is at index 15

        self.left_gripper_signal.value = l_gripper_cmd * 1e3
        self.right_gripper_signal.value = r_gripper_cmd * 1e3

        super().update_target_joints(
        joints=mapped_action,
        enable=True
        )
        
        ######################################################################
        
        self.update_visualization()
        super().publish_joint_commands()
        
        if rate is not None:
            rate.sleep()
    
    def _map_action_to_robot_joints(self, action):
        """
        Map the action from the model's format to the robot's joint order.
        
        Model format (action):
        yumi_joint_1_l, yumi_joint_1_r, yumi_joint_2_l, yumi_joint_2_r, 
        yumi_joint_7_l, yumi_joint_7_r, yumi_joint_3_l, yumi_joint_3_r,
        yumi_joint_4_l, yumi_joint_4_r, yumi_joint_5_l, yumi_joint_5_r,
        yumi_joint_6_l, yumi_joint_6_r, gripper_l_joint, gripper_r_joint
        
        Robot format (self.joints):
        yumi_joint_1_r, yumi_joint_2_r, yumi_joint_7_r, yumi_joint_3_r,
        yumi_joint_4_r, yumi_joint_5_r, yumi_joint_6_r, yumi_joint_1_l,
        yumi_joint_2_l, yumi_joint_7_l, yumi_joint_3_l, yumi_joint_4_l,
        yumi_joint_5_l, yumi_joint_6_l, gripper_r_joint, gripper_l_joint
        """
        # Create a new array with the correct order
        mapped_action = onp.zeros(16)
        
        # Map right arm joints
        mapped_action[0] = action[1]  # yumi_joint_1_r
        mapped_action[1] = action[3]  # yumi_joint_2_r
        mapped_action[2] = action[5]  # yumi_joint_7_r
        mapped_action[3] = action[7]  # yumi_joint_3_r
        mapped_action[4] = action[9]  # yumi_joint_4_r
        mapped_action[5] = action[11] # yumi_joint_5_r
        mapped_action[6] = action[13] # yumi_joint_6_r
        
        # Map left arm joints
        mapped_action[7] = action[0]  # yumi_joint_1_l
        mapped_action[8] = action[2]  # yumi_joint_2_l
        mapped_action[9] = action[4]  # yumi_joint_7_l
        mapped_action[10] = action[6] # yumi_joint_3_l
        mapped_action[11] = action[8] # yumi_joint_4_l
        mapped_action[12] = action[10] # yumi_joint_5_l
        mapped_action[13] = action[12] # yumi_joint_6_l
        
        # Map gripper joints
        mapped_action[14] = action[15] # gripper_r_joint
        mapped_action[15] = action[14] # gripper_l_joint
        
        return mapped_action
    
    def update_curr_proprio(self):
        """
        Update the current proprioception state by mapping the robot's joint order to the model's format.
        
        Robot format (self.joints):
        yumi_joint_1_r, yumi_joint_2_r, yumi_joint_7_r, yumi_joint_3_r,
        yumi_joint_4_r, yumi_joint_5_r, yumi_joint_6_r, yumi_joint_1_l,
        yumi_joint_2_l, yumi_joint_7_l, yumi_joint_3_l, yumi_joint_4_l,
        yumi_joint_5_l, yumi_joint_6_l, gripper_r_joint, gripper_l_joint
        
        Model format (self.cur_proprio):
        yumi_joint_1_l, yumi_joint_1_r, yumi_joint_2_l, yumi_joint_2_r, 
        yumi_joint_7_l, yumi_joint_7_r, yumi_joint_3_l, yumi_joint_3_r,
        yumi_joint_4_l, yumi_joint_4_r, yumi_joint_5_l, yumi_joint_5_r,
        yumi_joint_6_l, yumi_joint_6_r, gripper_l_joint, gripper_r_joint
        """
        # Create a new array with the correct order
        self.cur_proprio = onp.zeros(16)
        
        # Map left arm joints
        self.cur_proprio[0] = self.joints[7]  # yumi_joint_1_l
        self.cur_proprio[2] = self.joints[8]  # yumi_joint_2_l
        self.cur_proprio[4] = self.joints[9]  # yumi_joint_7_l
        self.cur_proprio[6] = self.joints[10] # yumi_joint_3_l
        self.cur_proprio[8] = self.joints[11] # yumi_joint_4_l
        self.cur_proprio[10] = self.joints[12] # yumi_joint_5_l
        self.cur_proprio[12] = self.joints[13] # yumi_joint_6_l
        
        # Map right arm joints
        self.cur_proprio[1] = self.joints[0]  # yumi_joint_1_r
        self.cur_proprio[3] = self.joints[1]  # yumi_joint_2_r
        self.cur_proprio[5] = self.joints[2]  # yumi_joint_7_r
        self.cur_proprio[7] = self.joints[3]  # yumi_joint_3_r
        self.cur_proprio[9] = self.joints[4]  # yumi_joint_4_r
        self.cur_proprio[11] = self.joints[5] # yumi_joint_5_r
        self.cur_proprio[13] = self.joints[6] # yumi_joint_6_r
        
        # Map gripper joints
        self.cur_proprio[14] = self.joints[15] # gripper_l_joint
        self.cur_proprio[15] = self.joints[14] # gripper_r_joint
        
        assert self.cur_proprio.shape == (16,)

        self.proprio_buffer.append(self.cur_proprio)

    def episode_start(self):
        """Reset the environment and start a new episode."""
        self.image_primary.clear()
        self.image_wrist.clear()
        self.prev_action.clear()
        self.proprio_buffer.clear()
        self.action_queue.clear()
        self.cur_proprio = None
        self.home() # Move to home position as first action
        rospy.sleep(5)
    
    def _setup_scene(self):
        # if self.control_mode == 'receding_horizon_control':
        while self.cartesian_pose_L is None or self.cartesian_pose_R is None:
            rospy.sleep(0.1)
        self.action_queue_viz_L = self.server.scene.add_point_cloud(
            name = "action_queue_L", 
            points = onp.array([[self.cartesian_pose_L.transform.translation.x,
                                self.cartesian_pose_L.transform.translation.y,
                                self.cartesian_pose_L.transform.translation.z]]), 
            colors = onp.array([[1.0, 0.0, 0.0]]), 
            point_size=0.002,
            point_shape='circle'
            )
        self.action_queue_viz_R = self.server.scene.add_point_cloud(
            name = "action_queue_R", 
            points = onp.array([[self.cartesian_pose_R.transform.translation.x,
                                self.cartesian_pose_R.transform.translation.y,
                                self.cartesian_pose_R.transform.translation.z]]), 
            colors = onp.array([[1.0, 0.0, 0.0]]), 
            point_size=0.002,
            point_shape='circle'
            )
        self.proprio_queue_viz_L = self.server.scene.add_point_cloud(
            name = "proprio_queue_L", 
            points = onp.array([[self.cartesian_pose_L.transform.translation.x,
                                self.cartesian_pose_L.transform.translation.y,
                                self.cartesian_pose_L.transform.translation.z]]), 
            colors = onp.array([[1.0, 0.0, 1.0]]), 
            point_size=0.003,
            )
        self.proprio_queue_viz_R = self.server.scene.add_point_cloud(
            name = "proprio_queue_R", 
            points = onp.array([[self.cartesian_pose_R.transform.translation.x,
                                self.cartesian_pose_R.transform.translation.y,
                                self.cartesian_pose_R.transform.translation.z]]), 
            colors = onp.array([[1.0, 0.0, 1.0]]), 
            point_size=0.003,
            )
            
    def _update_action_queue_viz(self):
        action_queue_L = self.action_prediction[0,:,:3]
        action_queue_R = self.action_prediction[0,:,10:13]
        self.action_queue_viz_L.points = action_queue_L
        color_order = onp.linspace(0.0, 1.0, len(action_queue_L))
        self.action_queue_viz_L.colors = onp.array([onp.array([c, 0.0, 0.0]) for c in color_order])
        self.action_queue_viz_R.points = action_queue_R
        self.action_queue_viz_R.colors = onp.array([onp.array([c, 0.0, 0.0]) for c in color_order])
    
    def _update_proprio_queue_viz(self):
        if len(self.proprio_buffer) > 0:
            proprio_queue_L = onp.array([a[:3] for a in onp.array(self.proprio_buffer)])
            proprio_queue_R = onp.array([a[10:13] for a in onp.array(self.proprio_buffer)])
            self.proprio_queue_viz_L.points = proprio_queue_L
            color_order = onp.linspace(0.0, 1.0, len(proprio_queue_L))
            self.proprio_queue_viz_L.colors = onp.array([onp.array([c, 0.0, 1.0]) for c in color_order])
            self.proprio_queue_viz_R.points = proprio_queue_R
            self.proprio_queue_viz_R.colors = onp.array([onp.array([c, 0.0, 1.0]) for c in color_order])
        
    def profile_run(self):
        profiler = LineProfiler()
        profiled_run = profiler(self.run)
        profiled_run()
        profiler.print_stats()

    def plot_predictions(self, input_proprio, action_prediction, timestamp):
        """
        Plot proprio history and predicted actions with meaningful labels.
        Args:
            input_proprio: input proprio data [B, T, D]
            action_prediction: predicted actions [B, H, D] where H is horizon
        """
        
        labels = [
        # Left arm (0-9)
        'X_Left', 'Y_Left', 'Z_Left',
        'Rot1_Left', 'Rot2_Left', 'Rot3_Left', 'Rot4_Left', 'Rot5_Left', 'Rot6_Left',
        'Grip_Left',
        # Right arm (10-19)
        'X_Right', 'Y_Right', 'Z_Right',
        'Rot1_Right', 'Rot2_Right', 'Rot3_Right', 'Rot4_Right', 'Rot5_Right', 'Rot6_Right',
        'Grip_Right'
        ]
        T = input_proprio.shape[1]  # Length of proprio history
        D = input_proprio.shape[2]  # Dimension of proprio/action
        H = action_prediction.shape[1]  # Prediction horizon
                
        fig, axes = plt.subplots(5, 4, figsize=(20, 20))
        plt.suptitle(f'Proprio History and Predictions - {timestamp}')
        
        for i in range(D):
            ax = axes[i//4, i%4]
            
            # Plot proprio history
            ax.plot(range(T), 
                    onp.flip(input_proprio[0, :, i].numpy(), 0), 
                    label='proprio', 
                    color='green')
            
            # Plot action predictions
            ax.plot(range(T-1, T+H-1),
                    action_prediction[0, :, i],
                    label='pred', 
                    color='red')
            ax.set_title(labels[i])
            ax.legend()
            ax.grid(True) 
        
        plt.tight_layout()

        os.makedirs(f'debug/{timestamp}', exist_ok=True)
        save_path = f'debug/{timestamp}/prediction_plot.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Saved prediction plot to {save_path}')


    def plot_stacked_obs(self, stacked_obs, timestamp):
        """
        Plot stacked observations from multiple cameras across time steps and save with timestamp.
        Args:
            stacked_obs: numpy array of shape [T, N_C, C, H, W]
                T: number of timesteps
                N_C: number of cameras
                C: channels (3 for RGB)
                H, W: height and width
            base_path: base name for the saved file (without extension)
        """
        
        save_path = f'debug/{timestamp}/stacked_obs.png'
        os.makedirs(f'debug/{timestamp}', exist_ok=True)
        
        T, N_C, C, H, W = stacked_obs.shape
        fig, axes = plt.subplots(T, N_C, figsize=(4*N_C, 4*T))
        
        if T == 1 and N_C == 1:
            axes = onp.array([[axes]])
        elif T == 1:
            axes = axes.reshape(1, -1)
        elif N_C == 1:
            axes = axes.reshape(-1, 1)
        
        plt.suptitle(f'Observation Stack - {timestamp}', y=1.02)
        
        for t in range(T):
            for nc in range(N_C):
                img = stacked_obs[t, nc].transpose(1, 2, 0)
                axes[t, nc].imshow(img)
                axes[t, nc].axis('off')
                axes[t, nc].set_title(f'Time {t}, Camera {nc}')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close()
        print(f'Saved plot to {save_path}')

    def add_gui_data_collection_controls(self):
        with self.server.gui.add_folder("Data Collection Controls"):
            self.start_record_button = self.server.gui.add_button("Start Recording")
            self.save_success_button = self.server.gui.add_button("Save Success", disabled=True)
            self.save_failure_button = self.server.gui.add_button("Save Failure", disabled=True)
            
        @self.start_record_button.on_click
        def _(_) -> None:
            """Callback for start recording."""
            self.start_record()
            self.start_record_button.disabled = True
            self.save_success_button.disabled = False
            self.save_failure_button.disabled = False
        @self.save_success_button.on_click
        def _(_) -> None:
            """Callback for save success."""
            self.save_success()
            self.start_record_button.disabled = False
            self.save_success_button.disabled = True
            self.save_failure_button.disabled = True
        @self.save_failure_button.on_click
        def _(_) -> None:
            """Callback for save failure."""
            self.save_failure()
            self.start_record_button.disabled = False
            self.save_success_button.disabled = True
            self.save_failure_button.disabled = True
    
    def _setup_collectors(self):
        if self.collect_data:
            self.start_record = rospy.ServiceProxy("/yumi_controller/start_recording", Empty)
            self.save_success = rospy.ServiceProxy("/yumi_controller/save_success", Empty)
            self.save_failure = rospy.ServiceProxy("/yumi_controller/save_failure", Empty)
            self.stop_record = rospy.ServiceProxy("/yumi_controller/stop_recording", Empty)

       
def main(
    ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/pi0_fast_yumi_finetune/", 
    ckpt_id: int = 29999,

    collect_data: bool = False,
    debug_mode: bool = True,
    task_name : str = 'put the white cup on the coffee machine',
    ): 
    
    yumi_interface = YuMiPI0PolicyController(ckpt_path, ckpt_id, task_name, collect_data, debug_mode)

    if collect_data:
        logger.info("Start data collection service")
        data_collector = DataCollector(init_node=False, task_name=task_name)

    yumi_interface.run()
    # yumi_interface.profile_run()
    
if __name__ == "__main__":
    tyro.cli(main)
