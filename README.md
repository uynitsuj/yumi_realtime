# YuMi Realtime Control Interfaces

A collection of realtime control interfaces for the bi-manual ABB YuMi IRB14000 robot arm using Robot Web Services (RWS) and Externally Guided Motion (EGM) low-level control.

Differential IK solving handled by the [pyroki](https://github.com/chungmin99/pyroki.git) library developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

Available control interfaces include:
1. Interactive Viser Interface (Draggable End-Effector Frames)
2. Meta Quest Oculus VR Tele-operation control
3. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
4. [Physical Intelligence PI0-FAST](https://www.physicalintelligence.company/research/fast)
5. [NVIDIA Isaac-GR00T](https://developer.nvidia.com/isaac/gr00t)

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
- [x] Implement [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) Controller
- [x] Implement [NVIDIA Isaac-GR00T](https://developer.nvidia.com/isaac/gr00t) Policy Controller
- [x] Implement [Physical Intelligence π0](https://www.physicalintelligence.company/blog/pi0) Policy Controller
