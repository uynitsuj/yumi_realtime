# YuMi Realtime Control Interfaces

A collection of realtime control interfaces for the bi-manual ABB YuMi IRB14000 robot arm using Robot Web Services (RWS) and Externally Guided Motion (EGM) low-level control.

Factor-graph-based inverse-kinematics optimization solver handled by the [jaxmp](https://github.com/chungmin99/jaxmp) library developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

Available control interfaces include:
1. Interactive Viser Interface (Draggable End-Effector Frames)
2. Meta Quest Oculus VR Tele-operation control
3. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

![DiffusionPolicy](data/media/DiffusionPolicyHandoffYuMi.gif)
Diffusion Policy autonomously performing a handoff trained from 70 expert demonstrations

![VR_Control](data/media/YuMiTeleopVR.gif)
Oculus VR Controllers Tele-operating the YuMi

## Installation
Full [install instructions](INSTALL.md) tested on Ubuntu 22.04 ROS Noetic in a mamba-forge environment.

## Usage Instructions
After a successful install, read the [usage instructions](USAGE_GUIDE.md).

## TODOS

- [x] Add observation-action pair .hdf5 logger for online visuo-motor policy training.
- [x] Add data playback visualizer for data logging verification
- [x] Bind datalogger to oculus save/reject data buttons
- [x] Debug VRPolicy action behavior while homing
- [x] Debug frame twitching in visualization
- [x] Implement Diffusion Policy Controller
