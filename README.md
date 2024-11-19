# YuMi Realtime Control Interfaces

A collection of realtime control interfaces for the bi-manual ABB YuMi IRB14000 robot arm using Robot Web Services (RWS) and Externally Guided Motion (EGM) low-level control.

Available control interfaces include:
1. Interactive Viser Interface (Draggable End-Effector Frames)
2. Meta Quest Oculus VR Tele-operation control

![VR_Control](data/media/YuMiTeleopVR.gif)
Oculus VR Controllers Tele-operating the YuMi

## Installation
Full [install instructions](INSTALL.md) tested on Ubuntu 22.04 ROS Noetic in a mamba-forge environment.

## Usage Instructions
After a successful install, read the [usage instructions](USAGE_GUIDE.md).

## TODOS

- [x] Add observation-action pair .hdf5 logger for online visuo-motor policy training.
- [x] Add data playback visualizer for data logging verification
- [ ] Bind datalogger to oculus save/reject data buttons
- [ ] Debug frame twitching in visualization
