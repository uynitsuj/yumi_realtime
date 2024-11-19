# Installation
Full install tested on Ubuntu 22.04 ROS Noetic in mamba-forge environment.

```
cd ~/
git clone --recurse-submodules https://github.com/uynitsuj/yumi_realtime.git
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

## Create Mamba-Forge environment
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

## Install yumi_realtime
```bash
cd ~/yumi_realtime
python -m pip install -e .
```

## Install ROS Noetic into the environment
```bash
mamba install ros-noetic-desktop
mamba deactivate
mamba activate yumiegmros

mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

### Test mamba-forge ros installation
```bash
mamba activate yumiegmros
roscore
```
Should start a roscore instance

## Install ABB Robot Driver and dependencies

### Building the Packages

```bash
# Change to the root of the Catkin workspace.
cd ~/yumi_realtime/catkin_ws/src

git clone https://github.com/ros-industrial/abb_robot_driver_interfaces.git

mamba install protobuf ros-noetic-controller-manager ros-noetic-joint-state-controller ros-noetic-velocity-controllers ros-noetic-position-controllers ros-noetic-controller-manager-msgs ros-noetic-hardware-interface ros-noetic-joint-limits-interface ros-noetic-controller-interface ros-noetic-realtime-tools

# Finally build the workspace (may take a minute)
cd ~/yumi_realtime/catkin_ws
catkin build
```
Clean install should result in no error messages

Finally, activate the workspace to get access to the packages just built:
```bash
source ~/yumi_realtime/catkin_ws/devel/setup.bash
```

## Test YuMi launch
```bash
roslaunch abb_robot_bringup yumi_robot.launch
```

### Install Issues
If linking issues with libabb_libegm.so or libprotobuf.so occur try adding to the launch file with something like...
```bash
<?xml version="1.0"?>
<launch>

  # export paths to shared libraries in 
  <env name="LD_LIBRARY_PATH" value="/home/<user>/catkin_ws/devel/abb_libegm/lib:/home/<user>/miniforge3/envs/yumiegmros/lib:${LD_LIBRARY_PATH}" />
  <arg name="robot_ip" doc="The robot controller's IP address"/>
```

