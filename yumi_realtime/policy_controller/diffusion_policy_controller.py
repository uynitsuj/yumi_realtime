from yumi_realtime.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
import rospy
from typing import Literal
from yumi_realtime.base import YuMiBaseInterface
from yumi_realtime.policy_controller.utils.utils import *
# from dp_gs.dataset.utils import *
from dp_gs.policy.diffusion_wrapper import DiffusionWrapper
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
import signal
import sys

class YuMiDiffusionPolicyController(YuMiROSInterface):
    """YuMi controller for diffusion policy control."""
    
    def __init__(self, ckpt_path: str = None, ckpt_id: int = 0, collect_data: bool = False, debug_mode = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_handles = False
        
        for side in ['left', 'right']:
            target_handle = self.transform_handles[side]
            target_handle.control.visible = False
        
        assert ckpt_path is not None, "Diffusion Policy checkpoint path must be provided."
        
        # Setup Diffusion Policy module and weights
        self.model = DiffusionWrapper(model_ckpt_folder=ckpt_path, ckpt_id=ckpt_id, device='cuda')
        self.collect_data = collect_data

        # ROS Camera Observation Subscriber
        self.bridge = CvBridge()
        self.height = None
        self.width = None
        
        self.proprio_buffer = deque([],maxlen=self.model.model.obs_horizon)

        self.observation_buffers = {}
        self.camera_buffers = {}
        self.camera_topics = [topic[0] for topic in rospy.get_published_topics() if 'sensor_msgs/Image' in topic[1]]
        
        self.relative_action_mode = False
        self.shutting_down = False
        
        # Initialize a deque for each camera
        self.max_buffer_size = 5  # For storing recent messages to sync from
        self.main_camera = "camera_0"
        for idx, topic in enumerate(self.camera_topics):
            camera_name = f"camera_{topic.split('camera_')[1][0]}"
            if camera_name == self.main_camera:
                # Main camera's synchronized buffer
                self.observation_buffers[camera_name] = deque([], maxlen=self.model.model.obs_horizon)
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
        
        self.cartesian_pose_L = None
        self.cartesian_pose_R = None
        
        # Control mode
        # self.control_mode = 'receding_horizon_control'
        self.control_mode = 'temporal_ensemble'
        
        self.skip_actions = 6 # 6 For DP
        # self.skip_actions = 4 # drawer sim 1k
        # self.skip_actions = 0
        # self.skip_actions = 4
        self.skip_every_other_pred = False
        if self.control_mode == 'receding_horizon_control':
            # self.max_len = self.model.model.action_horizon//2
            self.max_len = 16
            self.action_queue = deque([],maxlen=self.max_len)
        elif self.control_mode == 'temporal_ensemble':
            if self.skip_every_other_pred:
                self.action_queue = deque([],maxlen=self.model.model.action_horizon//3 - self.skip_actions)
            else:
                self.action_queue = deque([],maxlen=self.model.model.action_horizon - self.skip_actions)
        self.prev_action = deque([],maxlen=self.model.model.obs_horizon)
        
        self._setup_scene()
        
        logger.info("Diffusion Policy controller initialized")

        if self.collect_data:
            self._setup_collectors()
            self.add_gui_data_collection_controls()

        # self.gripper_thres = 0.59
        # self.gripper_thres = 0.69
        self.gripper_thres = 0.5
        # self.gripper_thres = 0.15
        # self.gripper_thres = 0.90

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
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal by homing the robot and resetting."""
        if self.shutting_down:
            return
            
        self.shutting_down = True
        logger.info('\nCtrl+C detected. Homing robot and resetting...')
        
        try:
            # Home the robot
            start = time.time()
            i = 0
            
            logger.info("Pre-Homing...")
            while ((self.height is None or self.width is None) 
                or time.time() - start < 2
                or len(self.proprio_buffer) != self.model.model.obs_horizon):
                self.pre_home()
                if self.rate is not None:
                    self.rate.sleep()
                self.solve_ik()
                self.update_visualization()
                super().publish_joint_commands()
                if i % 300 == 0:
                    self.call_gripper(side = 'left', gripper_state = False, enable = True)
                    rospy.sleep(0.2)
                    self.call_gripper(side = 'right', gripper_state = False, enable = True)
                    i = 0
                i += 1
            start = time.time()
            logger.info("Homing...")

            self.proprio_buffer.clear()
            for key in self.observation_buffers.keys():
                self.observation_buffers[key].clear()
            self.last_action = None
            self.action_queue.clear()
            
            while ((self.height is None or self.width is None) 
                or time.time() - start < 4
                or len(self.proprio_buffer) != self.model.model.obs_horizon):
                self.home()
                if self.rate is not None:
                    self.rate.sleep()
                self.solve_ik()
                self.update_visualization()
                
                super().publish_joint_commands()
                if i % 400 == 0:
                    self.call_gripper(side = 'left', gripper_state = False, enable = True)
                    rospy.sleep(0.2)
                    self.call_gripper(side = 'right', gripper_state = False, enable = True)
                    i = 0
                i += 1

            # Prealloc diffusion input tensors
            camera_count = len(self.observation_buffers.keys())
            self.input_obs_tensor = torch.empty(
                (1, self.model.model.obs_horizon, camera_count, 3, self.height, self.width),
                dtype=torch.float32,
                device='cpu'
            )

            action_dim = len(self.proprio_buffer[0])
            self.input_proprio_tensor = torch.empty(
                (1, self.model.model.obs_horizon, action_dim),  # Assuming 20 is the proprio dimension
                dtype=torch.float32,
                device='cpu'
            )
            # If we're collecting data, stop recording
            if self.collect_data:
                try:
                    self.stop_record()
                    print("Data recording stopped.")
                except Exception as e:
                    print(f"Error stopping data recording: {e}")
            

            # logger.info("Press C to continue once finished homing")
            # import pdb; pdb.set_trace()
            self.shutting_down = False
            self.run()

        except Exception as e:
            print(f"Error during shutdown: {e}")

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
        
        if self.cartesian_pose_L is None or self.cartesian_pose_R is None:
            logger.info("fail to read the cartesian pose")
            return

        target_time = image_msg.header.stamp.to_nsec()
        
        # Process main camera
        onp_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8').astype("float32") / 255.0
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
            onp_img = self.bridge.imgmsg_to_cv2(closest_msg, desired_encoding='rgb8').astype("float32") / 255.0
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
                    self.observation_buffers[camera_name] = deque([obs] * (self.model.model.obs_horizon - 1), 
                                                        maxlen=self.model.model.obs_horizon)
                self.observation_buffers[camera_name].append(obs)

    def run(self):
        """Diffusion Policy controller loop."""
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        logger.info("Running rollout... press Ctrl+C once to stop early, twice to home+reset.")

        self.rate = rospy.Rate(150) # 150Hz control loop
        self.rate = None          
        # self.home()
        i = 0
        while (self.cartesian_pose_L is None or self.cartesian_pose_R is None or not self.has_jitted):
            self.solve_ik()
            time.sleep(0.1)

        start = time.time()
        logger.info("Pre-Homing...")
        while ((self.height is None or self.width is None) 
                or time.time() - start < 2
                or len(self.proprio_buffer) != self.model.model.obs_horizon):
            self.pre_home()
            if self.rate is not None:
                self.rate.sleep()
            self.solve_ik()
            self.update_visualization()
            super().publish_joint_commands()
            if i % 300 == 0:
                self.call_gripper(side = 'left', gripper_state = False, enable = True)
                rospy.sleep(0.2)
                self.call_gripper(side = 'right', gripper_state = False, enable = True)
                i = 0
            i += 1
        start = time.time()

        logger.info("Homing...")
        while ((self.height is None or self.width is None) 
               or time.time() - start < 4
               or len(self.proprio_buffer) != self.model.model.obs_horizon):
            self.home()
            if self.rate is not None:
                self.rate.sleep()
            self.solve_ik()
            self.update_visualization()
            
            super().publish_joint_commands()
            if i % 300 == 0:
                self.call_gripper(side = 'left', gripper_state = False, enable = True)
                rospy.sleep(0.2)
                self.call_gripper(side = 'right', gripper_state = False, enable = True)
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

        # Prealloc diffusion input tensors
        camera_count = len(self.observation_buffers.keys())
        self.input_obs_tensor = torch.empty(
            (1, self.model.model.obs_horizon, camera_count, 3, self.height, self.width),
            dtype=torch.float32,
            device='cpu'
        )

        action_dim = len(self.proprio_buffer[0])
        self.input_proprio_tensor = torch.empty(
            (1, self.model.model.obs_horizon, action_dim),  # Assuming 20 is the proprio dimension
            dtype=torch.float32,
            device='cpu'
        )

        step = 0
        self.last_action = None
        while not rospy.is_shutdown():
            assert len(self.proprio_buffer) == self.model.model.obs_horizon
            
            if self.debug_mode:
                _img_0 = (stacked_obs[-1][0].transpose([1,2,0])*255).astype(onp.uint8)
                _img_1 = (stacked_obs[-1][1].transpose([1,2,0])*255).astype(onp.uint8)
                list(self.viser_img_handles.items())[0][1].image = _img_0
                list(self.viser_img_handles.items())[1][1].image = _img_1
            
            self._update_proprio_queue_viz()
            
            start = time.time()
            step += 1

            if self.last_action is not None:
                # check gripper state of last action 
                target_left_gripper = self.last_action[9] < self.gripper_thres
                target_right_gripper = self.last_action[19] < self.gripper_thres

                current_left_gripper = input["proprio"][0, -1, 9] < self.gripper_thres
                current_right_gripper = input["proprio"][0, -1, 19] < self.gripper_thres
                
                print("current_left_gripper: ", current_left_gripper)
                print("current_right_gripper: ", current_right_gripper)


                # delta pose 
                target_proprio_left = self.last_action[:3]
                current_proprio_left = input["proprio"][0, -1, :3].numpy()
            
                # calculate lag 
                xyz_lag = onp.linalg.norm(target_proprio_left - current_proprio_left)
                print("lag: ", xyz_lag)

                # if they are in disgreemnt, gripper control with last action 
                # if target_left_gripper != current_left_gripper or target_right_gripper != current_right_gripper:
                if target_left_gripper != current_left_gripper or target_right_gripper != current_right_gripper or xyz_lag > 0.0020:
                    print("blocking with last action")
                    self._yumi_control(self.last_action)
                    self.last_action = None
                    continue

            # receding horizon control
            if self.control_mode == 'receding_horizon_control':
                if len(self.action_queue) > 0:
                    action = self.action_queue.popleft()
                    self._yumi_control(action)
                    continue
            # end of receding horizon control

            all_camera_obs = []
            for camera_name in self.observation_buffers.keys():
                assert len(self.observation_buffers[camera_name]) == self.model.model.obs_horizon
                cam_obs = onp.array(self.observation_buffers[camera_name])
                all_camera_obs.append(cam_obs)
            
            # Stack along the N_C dimension
            stacked_obs = onp.stack(all_camera_obs, axis=1)  # [T, N_C, C, H, W]
            self.input_obs_tensor.copy_(torch.from_numpy(stacked_obs).unsqueeze(0))
            
            proprio_array = onp.asarray(self.proprio_buffer)
            self.input_proprio_tensor.copy_(torch.flip(torch.from_numpy(proprio_array).unsqueeze(0), [1]))
            # self.input_proprio_tensor.copy_(torch.from_numpy(proprio_array).unsqueeze(0))

            # Create input dict referencing pre-allocated tensors
            input = {
                "observation": self.input_obs_tensor,
                "proprio": self.input_proprio_tensor
            }

            inference_start = time.time()
            print("input proprio gripper: ", input["proprio"][..., -1])
            action_prediction = self.model(input) # Denoise action prediction from obs and proprio...

            # debug gripper (Max)
            left_gripper = action_prediction[0, :, 9]
            # right_gripper = action_prediction[0, :, -1]
            print("All left gripper prediction: ", left_gripper)
            # end debug

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
            self._update_action_queue_viz()

            # Handle relative vs absolute action predictions
            if hasattr(self, 'relative_action_mode') and self.relative_action_mode:
                # import pdb; pdb.set_trace()
                # action_L = convert_abs_action(action_prediction[:,:,:10], self.cur_proprio[:10][None,None])[0]
                action_L = action_prediction[0,:,:10]
                action_R = convert_abs_action(action_prediction[:,:,10:], self.cur_proprio[10:][None,None])[0]
            else:
                action_L = action_prediction[0,:,:10]
                action_R = action_prediction[0,:,10:]
            
            action = onp.concatenate([action_L, action_R], axis=-1)

            # # temporal emsemble start
            if self.control_mode == 'temporal_ensemble':
                if self.skip_every_other_pred:
                    action_prediction = action_prediction[::3]
                new_actions = deque(action[self.skip_actions:self.model.model.action_horizon])
                self.action_queue.append(new_actions)
                if self.model.model.pred_left_only or self.model.model.pred_right_only:
                    actions_current_timestep = onp.empty((len(self.action_queue), self.model.model.action_dim*2))
                else:
                    actions_current_timestep = onp.empty((len(self.action_queue), self.model.model.action_dim))
                
                k = 0.005
                # k = 0.05
                for i, q in enumerate(self.action_queue):
                    actions_current_timestep[i] = q.popleft()

                exp_weights = onp.exp(k * onp.arange(actions_current_timestep.shape[0]))
                exp_weights = exp_weights / exp_weights.sum()

                action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
                self.temporal_ensemble_action = action
                # action[9] = action_L[2,-1]
                # action[19] = action_R[2,-1]

                
            # receding horizon # check the receding horizon block as well
            if self.control_mode == 'receding_horizon_control':
                if len(self.action_queue) == self.max_len: # If at max, discard a few due to latency
                    for i in range(self.skip_actions):
                        self.action_queue.popleft()
                        
                if len(self.action_queue) == 0: 
                    self.action_queue = deque([a for a in action[:self.max_len]])
                action = self.action_queue.popleft()
            
            # update yumi action 
            self._yumi_control(action)
            # self._update_action_queue_viz()
    
    def _yumi_control(self, action):
        # YuMi action update
        ######################################################################
        print("action update called")
        self.last_action = action
        l_act = action_10d_to_8d(action[:10])
        r_act = action_10d_to_8d(action[10:])
        l_xyz, l_wxyz, l_gripper_cmd = l_act[:3], l_act[3:-1], l_act[-1]
        r_xyz, r_wxyz, r_gripper_cmd = r_act[:3], r_act[3:-1], r_act[-1]

        if l_xyz[2] < 0.005:
            l_xyz[2] = 0.006
            # import pdb; pdb.set_trace()
        if r_xyz[2] < 0.005:
            r_xyz[2] = 0.006
            # import pdb; pdb.set_trace()
        print("left xyz: ", l_xyz)
        print("left gripper: ", l_gripper_cmd)
        print("right xyz: ", r_xyz)
        print("right gripper: ", r_gripper_cmd)

        self.left_gripper_signal.value = l_gripper_cmd * 1e3
        self.right_gripper_signal.value = r_gripper_cmd * 1e3

        super().update_target_pose(
        side='left',
        position=l_xyz,
        wxyz=l_wxyz,
        gripper_state=bool(l_gripper_cmd>self.gripper_thres), 
        enable=True if not self.model.model.pred_right_only else False
        )
        
        super().update_target_pose(
        side='right',
        position=r_xyz,
        wxyz=r_wxyz,
        gripper_state=bool(r_gripper_cmd>self.gripper_thres), 
        enable= True if not self.model.model.pred_left_only else False
        )
        ######################################################################
        
        self.solve_ik()
        self.update_visualization()
        super().publish_joint_commands()
        
        if self.rate is not None:
            self.rate.sleep()
    
    def update_curr_proprio(self):
        l_xyz, l_xyzw = tf2xyz_quat(self.cartesian_pose_L)
        r_xyz, r_xyzw = tf2xyz_quat(self.cartesian_pose_R)
        
        l_q = Rotation.from_quat(l_xyzw)
        l_rot = l_q.as_matrix()
        l_rot_6d = onp.squeeze(rot_mat_to_rot_6d(l_rot[None]), axis=0)# [N, 6]
        r_q = Rotation.from_quat(r_xyzw)
        r_rot = r_q.as_matrix()
        r_rot_6d = onp.squeeze(rot_mat_to_rot_6d(r_rot[None]), axis=0) # [N, 6]
        
        self.cur_proprio = onp.concatenate([l_xyz, l_rot_6d, onp.array([int(self.gripper_L_pos < 200)]), r_xyz, r_rot_6d, onp.array([int(self.gripper_R_pos) < 200])], axis=-1, dtype=onp.float32)
        assert self.cur_proprio.shape == (20,)

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
    # some signal 
    # ckpt_path : str = "/home/xi/checkpoints/241203_2002", 
    # ckpt_id: int = 99
    
    # Dec 4
    # ckpt_path : str = "/home/xi/checkpoints/241203_2104", 
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/checkpoints/241203_217", 
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/checkpoints/241203_2108", 
    # ckpt_id : int = 599,

    # ckpt_path : str = "/home/xi/checkpoints/241205_1219",
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/checkpoints/241205_1547",
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/yumi_realtime/dependencies/dp_gs/output/241223_1409_athena5", 
    # ckpt_id : int = 70,

    # ckpt_path : str = "/home/xi/Documents/dp_gs/out_dir/241211_1403_athena5", 
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/checkpoints/241205_1609",
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/yumi_realtime/dependencies/dp_gs/output/241230_2055_athena5",
    # ckpt_id : int = 299,

    # ckpt_path : str = "/home/xi/yumi_realtime/dependencies/dp_gs/output/250103_1706_sim_athena5",
    # ckpt_id : int = 299,

    # ckpt_path: str = "/home/xi/checkpoints/yumi_pick_tiger/250122_1721", # works!
    # ckpt_id: int = 340,

    # ckpt_path: str = "/home/xi/checkpoints/yumi_pick_tiger_bimanual/250123_2225", # first bimanual
    # ckpt_id: int = 490,

    # ckpt_path: str = "/home/xi/checkpoints/yumi_pick_tiger_bimanual/250124_1935",
    # ckpt_id: int = 400,

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250311_2134", # first datagen working, coffee maker no aug
    # ckpt_id: int = 15,

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250314_2126", 
    # ckpt_id: int = 100, #some signal on trajinterp

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250316_2247/", 
    # ckpt_id: int = 130, # more signal on trajinterp

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250316_2248_1hist/", 
    # ckpt_id: int = 275,

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250317_2121/", 
    # ckpt_id: int = 50, # DINO enc diffusion

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250316_1250/", 
    # ckpt_id: int = 5, 

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250319_1715/", 
    # ckpt_id: int = 299, #Diffusion working on coffee maker, 6 skip actions  0.5 gripper thres

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250409_2102/", 
    # ckpt_id: int = 5, # Diffusion 5k on coffee maker not working

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250409_2102_resume/", 
    # ckpt_id: int = 50, # Diffusion 5k on coffee maker

    # ckpt_path: str = "/mnt/spare-ssd/dpgs_checkpoints/250411_2234", 
    # ckpt_id: int = 32, # Diffusion EMA 5k on coffee maker

    # ckpt_path: str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1216", 
    # ckpt_id: int = 49, # Diffusion spatial softmax EMA 1k on coffee maker

    # ckpt_path: str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1216", 
    # ckpt_id: int = 49, # Diffusion spatial softmax EMA 1k on coffee maker

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1732",
    # ckpt_id: int = 49, # diffusion subsample 50
    
    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1210",
    # ckpt_id: int = 49, # diffusion subsample 100
    
    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1003",
    # ckpt_id: int = 49, # diffusion subsample 200
    
    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1209",
    # ckpt_id: int = 32, # diffusion 5k new coffee maker

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250414_1209_resume",
    # ckpt_id: int = 4, # diffusion 5k new coffee maker

    # ckpt_path: str = "/home/xi/checkpoints/yumi_coffee_maker/250408_1442", 
    # ckpt_id: int = 30, # Simple 1k coffee maker

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250419_1208",
    # ckpt_id: int = 49, # sim LED v2 1k Apr 19

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250418_1711",
    # ckpt_id: int = 35, # sim faucet v1 1k Apr 19

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250418_2304",
    # ckpt_id: int = 35, # sim tiger bimanual 1k Apr 19

    # ckpt_path: str = "/mnt/spare-ssd/dpgs_checkpoints/250419_1644",
    # ckpt_id: int = 49, # sim cardboard box lift bimanual 1k Apr 20

    # ckpt_path: str = "/mnt/spare-ssd/dpgs_checkpoints/250420_1253",
    # ckpt_id: int = 49, # sim drawer 1k Apr 20

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250420_1311",
    # ckpt_id: int = 49, # sim tiger v3 Apr 20

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250420_1757",
    # ckpt_id: int = 99, # faucet sim 500 skipped

    ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250420_1758",
    ckpt_id: int = 200, # faucet sim 150 # half way finished training

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250420_1759",
    # ckpt_id: int = 499, # faucet sim 100

    # ckpt_path : str = "/mnt/spare-ssd/dpgs_checkpoints/250420_1800",
    # ckpt_id: int = 499, # faucet sim 50

    collect_data: bool = False,
    # task_name : str = 'pick up the cardboard box'
    # task_name : str = 'open the drawer'
    task_name : str = 'pick up the tiger'
    ): 
    
    yumi_interface = YuMiDiffusionPolicyController(ckpt_path, ckpt_id, collect_data)

    if collect_data:
        logger.info("Start data collection service")
        data_collector = DataCollector(init_node=False, task_name=task_name)

    yumi_interface.run()
    # yumi_interface.profile_run()
    
if __name__ == "__main__":
    tyro.cli(main)
