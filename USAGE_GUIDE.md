# Usage Guide

## Pre-flight checks
If the below guide does not work, go through the following checks:
- [] The YuMi should have the correct modules (to be added to repo) loaded on the pendant with State-Machine Add-In 1.1 and EGM installed 
- [] Under `Control Panel > Configuration > Communication > Transmission Protocol > [ROB_L and ROB_R]` should have the correct IP configured. (Should be set with `Remote Address: <ip address of computer>`. The correct ip address can be found by typing `ip addr` in a shell terminal). Additionally, ROB_L should be set to remote port `6511` and ROB_R with `6512`
- [] The YuMi should be set to `Automatic` instead of `Manual`, `Motors On` instead of off, and all module scripts should be running

## Step 1: Source ROS workspace and launch the YuMi bringup script
```bash
cd ~/yumi_ros_noetic/
. catkin_ws/devel/setup.bash

roslaunch abb_robot_bringup yumi_robot.launch
```

Upon successful launch, there should be no red `ERROR` messages (warnings are fine). Running the following command in a separate terminal (that has also sourced the workspace with `. catkin_ws/devel/setup.bash`) should allow you to see realtime joint angle feedback from the robot sensors.

```bash
rostopic echo /yumi/rws/joint_states
```

# Step 2: Choose your desired control interface script and 
