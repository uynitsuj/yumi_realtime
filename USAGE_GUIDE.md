# Usage Guide

## Pre-flight checks
If the below guide does not work, go through the following checks:
- The YuMi should have the correct modules (to be added to repo) loaded on the pendant with State-Machine Add-In 1.1 and EGM installed 
- Under `Control Panel > Configuration > Communication > Transmission Protocol > [ROB_L and ROB_R]` should have the correct IP configured. This should be set to `Remote Address: <ip address of computer>`. The correct ip address can be found by typing `ip addr` in a shell terminal. Additionally, ROB_L should be set to remote port `6511` and ROB_R with `6512`
- The YuMi should be set to `Automatic` instead of `Manual`, `Motors On` instead of off, and all module scripts should be running

## Step 1: Source ROS workspace and launch the YuMi bringup script
```bash
conda activate yumiegmros
cd ~/yumi_realtime/
. catkin_ws/devel/setup.bash

roslaunch abb_robot_bringup yumi_robot.launch
```

Upon successful launch, there should be no red `ERROR` messages (warnings are fine). Running the following command in a separate terminal should allow you to see real-time joint angle feedback from the robot sensors -- the new terminal should also source the workspace with `. catkin_ws/devel/setup.bash`.

```bash
rostopic echo /yumi/rws/joint_states
```

## Step 2: Choose your desired control interface and run script
Activate a new terminal and activate the `yumiegmros` env and source the ROS workspace again
```bash
conda activate yumiegmros
cd ~/yumi_realtime/
. catkin_ws/devel/setup.bash
```

The following real-time control interfaces are currently supported by this repository:

### Viser Control

A web visual interface with interactive transform handles (similar to moveit's interactive interface but this can be visualized over remote SSH with Viser)

```bash
python ~/yumi_realtime/yumi_realtime/controller.py
```

### Oculus VR Controller Tele-Operation Control

Open 2 additional terminals for 1. Oculus input reader node 2. Viser live UI and robot controller

1. Oculus device reader node
```bash
python ~/yumi_realtime/yumi_realtime/oculus_controller/utils/oculus_node.py
```

2. Viser live UI and robot controller interface
```bash
python ~/yumi_realtime/yumi_realtime/oculus_controller/oculus_control.py
```