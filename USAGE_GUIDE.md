# Usage Guide

## Pre-flight checks
If the below guide does not work, go through the following checks:
- The YuMi should have the correct modules (to be added to repo) loaded on the pendant with State-Machine Add-In 1.1 and EGM installed 
- Under `Control Panel > Configuration > Communication > Transmission Protocol > [ROB_L and ROB_R]` should have the correct IP configured. This should be set to `Remote Address: <ip address of computer>`. The correct ip address can be found by typing `ip addr` in a shell terminal. Additionally, ROB_L should be set to remote port `6511` and ROB_R with `6512`
- The YuMi should be set to `Automatic` instead of `Manual`, `Motors On` instead of off, and all module scripts should be running

## Step 1: Source ROS workspace and launch the YuMi bringup script
```bash
cd ~/yumi_ros_noetic/
. catkin_ws/devel/setup.bash

roslaunch abb_robot_bringup yumi_robot.launch
```

Upon successful launch, there should be no red `ERROR` messages (warnings are fine). Running the following command in a separate terminal should allow you to see real-time joint angle feedback from the robot sensors -- the new terminal should also source the workspace with `. catkin_ws/devel/setup.bash`.

```bash
rostopic echo /yumi/rws/joint_states
```

## Step 2: Choose your desired control interface and run script
The following real-time control interfaces are currently supported by this repository:
- Viser Control

A web visual interface with Inverse Kinematics via [jaxmp](https://github.com/chungmin99/jaxmp) and interactive transform handles (similar to moveit's interface but this can be visualized over remote SSH with Viser)
- Oculus VR Controller Tele-Operation Control

(Work In Progress)


Activate a new terminal and source the ROS workspace again
```bash
cd ~/yumi_ros_noetic/
. catkin_ws/devel/setup.bash
```

Then run the control interface python script.