# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "abb_rapid_sm_addin_msgs: 7 messages, 4 services")

set(MSG_I_FLAGS "-Iabb_rapid_sm_addin_msgs:/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg;-Iabb_rapid_msgs:/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg;-Istd_msgs:/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(abb_rapid_sm_addin_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" "abb_rapid_msgs/LoadData:abb_rapid_msgs/WObjData:abb_rapid_msgs/ToolData:abb_rapid_msgs/Orient:abb_rapid_msgs/Pose:abb_rapid_msgs/Pos"
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" "abb_rapid_msgs/Pos:abb_rapid_msgs/Orient:abb_rapid_msgs/Pose"
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" "abb_rapid_msgs/LoadData:abb_rapid_msgs/WObjData:abb_rapid_msgs/ToolData:abb_rapid_sm_addin_msgs/EGMSetupSettings:abb_rapid_msgs/Orient:abb_rapid_sm_addin_msgs/EGMRunSettings:abb_rapid_msgs/Pose:abb_rapid_sm_addin_msgs/EGMActivateSettings:abb_rapid_msgs/Pos:abb_rapid_sm_addin_msgs/EGMStopSettings"
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" ""
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" ""
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" "std_msgs/Header:abb_rapid_sm_addin_msgs/StateMachineState"
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" ""
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" "abb_rapid_msgs/LoadData:abb_rapid_msgs/WObjData:abb_rapid_msgs/ToolData:abb_rapid_sm_addin_msgs/EGMSetupSettings:abb_rapid_msgs/Orient:abb_rapid_sm_addin_msgs/EGMRunSettings:abb_rapid_msgs/Pose:abb_rapid_sm_addin_msgs/EGMActivateSettings:abb_rapid_msgs/Pos:abb_rapid_sm_addin_msgs/EGMSettings:abb_rapid_sm_addin_msgs/EGMStopSettings"
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" "abb_rapid_msgs/LoadData:abb_rapid_msgs/WObjData:abb_rapid_msgs/ToolData:abb_rapid_sm_addin_msgs/EGMSetupSettings:abb_rapid_msgs/Orient:abb_rapid_sm_addin_msgs/EGMRunSettings:abb_rapid_msgs/Pose:abb_rapid_sm_addin_msgs/EGMActivateSettings:abb_rapid_msgs/Pos:abb_rapid_sm_addin_msgs/EGMSettings:abb_rapid_sm_addin_msgs/EGMStopSettings"
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" ""
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" NAME_WE)
add_custom_target(_abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_rapid_sm_addin_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Services
_generate_srv_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_cpp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Module File
_generate_module_cpp(abb_rapid_sm_addin_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(abb_rapid_sm_addin_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages abb_rapid_sm_addin_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_rapid_sm_addin_msgs_gencpp)
add_dependencies(abb_rapid_sm_addin_msgs_gencpp abb_rapid_sm_addin_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_rapid_sm_addin_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Services
_generate_srv_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_eus(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Module File
_generate_module_eus(abb_rapid_sm_addin_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(abb_rapid_sm_addin_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages abb_rapid_sm_addin_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_rapid_sm_addin_msgs_geneus)
add_dependencies(abb_rapid_sm_addin_msgs_geneus abb_rapid_sm_addin_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_rapid_sm_addin_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Services
_generate_srv_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_lisp(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Module File
_generate_module_lisp(abb_rapid_sm_addin_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(abb_rapid_sm_addin_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages abb_rapid_sm_addin_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_rapid_sm_addin_msgs_genlisp)
add_dependencies(abb_rapid_sm_addin_msgs_genlisp abb_rapid_sm_addin_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_rapid_sm_addin_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Services
_generate_srv_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_nodejs(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Module File
_generate_module_nodejs(abb_rapid_sm_addin_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(abb_rapid_sm_addin_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages abb_rapid_sm_addin_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_rapid_sm_addin_msgs_gennodejs)
add_dependencies(abb_rapid_sm_addin_msgs_gennodejs abb_rapid_sm_addin_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_rapid_sm_addin_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_msg_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Services
_generate_srv_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv"
  "${MSG_I_FLAGS}"
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)
_generate_srv_py(abb_rapid_sm_addin_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
)

### Generating Module File
_generate_module_py(abb_rapid_sm_addin_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(abb_rapid_sm_addin_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages abb_rapid_sm_addin_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMActivateSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMRunSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMSetupSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/EGMStopSettings.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/RuntimeState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/msg/StateMachineState.msg" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/GetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetEGMSettings.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetRAPIDRoutine.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_sm_addin_msgs/srv/SetSGCommand.srv" NAME_WE)
add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py _abb_rapid_sm_addin_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_rapid_sm_addin_msgs_genpy)
add_dependencies(abb_rapid_sm_addin_msgs_genpy abb_rapid_sm_addin_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_rapid_sm_addin_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET abb_rapid_msgs_generate_messages_cpp)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp abb_rapid_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_rapid_sm_addin_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET abb_rapid_msgs_generate_messages_eus)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus abb_rapid_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_rapid_sm_addin_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET abb_rapid_msgs_generate_messages_lisp)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp abb_rapid_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_rapid_sm_addin_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET abb_rapid_msgs_generate_messages_nodejs)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs abb_rapid_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs)
  install(CODE "execute_process(COMMAND \"/home/xi/miniforge3/envs/yumiegmros/bin/python3.11\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_rapid_sm_addin_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET abb_rapid_msgs_generate_messages_py)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py abb_rapid_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(abb_rapid_sm_addin_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
