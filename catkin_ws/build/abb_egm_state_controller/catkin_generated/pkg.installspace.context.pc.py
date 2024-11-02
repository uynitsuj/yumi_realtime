# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "abb_egm_hardware_interface;abb_egm_msgs;abb_robot_cpp_utilities;controller_interface;pluginlib;realtime_tools;roscpp".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-labb_egm_state_controller".split(';') if "-labb_egm_state_controller" != "" else []
PROJECT_NAME = "abb_egm_state_controller"
PROJECT_SPACE_DIR = "/home/xi/yumi_ros_noetic/catkin_ws/install"
PROJECT_VERSION = "0.5.0"
