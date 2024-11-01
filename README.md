# YuMi ROS Noetic Interface

## Installation
Full install tested on Ubuntu 22.04 ROS Noetic in mamba-forge environment.

```
cd ~/
git clone --recurse-submodules https://github.com/uynitsuj/yumi_ros_noetic.git
```

## Install Mamba-Forge
Install mamba-forge
Download the installer using curl or wget or your favorite program and run the script.
For eg:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
or
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

## create Mamba-Forge ROS noetic environment
```bash
mamba create -n yumiegmros python=3.11
mamba activate yumiegmros
```

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults
```
# Install ros-noetic into the environment (ROS1)
```bash
mamba install ros-noetic-desktop
mamba deactivate
mamba activate yumiegmros

mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

#### Test mamba-forge ros installation
```bash
mamba activate yumiegmros
roscore
```

## Install abb-robot-driver and dependencies

### Building the Packages

The following instructions should build the main branches of all required repositories on a ROS Melodic system:

```bash
# Change to the root of the Catkin workspace.
cd ~/yumi_ros_noetic/catkin_ws/src

# Check build dependencies.
# First update the local rosdep database.
rosdep update

git clone https://github.com/ros-industrial/abb_robot_driver_interfaces.git

mamba install protobuf ros-noetic-controller-manager ros-noetic-joint-state-controller ros-noetic-velocity-controllers ros-noetic-controller-manager-msgs ros-noetic-hardware-interface ros-noetic-joint-limits-interface ros-noetic-controller-interface ros-noetic-realtime-tools

# Finally build the workspace (may take a minute)
cd ~/yumi_ros_noetic/catkin_ws
catkin_make_isolated
```
Clean install should result in no error messages

Finally, activate the workspace to get access to the packages just built:
```bash
source ~/yumi_ros_noetic/catkin_ws/devel_isolated/setup.bash
```

### Test YuMi launch
```bash
roslaunch abb_robot_bringup yumi_robot.launch
```

### Install Issues
If linking issues with libabb_libegm.so or libprotobuf.so occur try adding to the launch file with something like...
```bash
<?xml version="1.0"?>
<launch>

  # export paths to shared libraries in 
  <env name="LD_LIBRARY_PATH" value="/home/<user>/catkin_ws/devel_isolated/abb_libegm/lib:/home/<user>/miniforge3/envs/yumiegmros/lib:${LD_LIBRARY_PATH}" />
  <arg name="robot_ip" doc="The robot controller's IP address"/>
```

