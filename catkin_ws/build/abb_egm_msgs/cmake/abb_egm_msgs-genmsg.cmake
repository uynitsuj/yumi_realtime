# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "abb_egm_msgs: 2 messages, 0 services")

set(MSG_I_FLAGS "-Iabb_egm_msgs:/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg;-Istd_msgs:/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(abb_egm_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" NAME_WE)
add_custom_target(_abb_egm_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_egm_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" ""
)

get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" NAME_WE)
add_custom_target(_abb_egm_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "abb_egm_msgs" "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" "std_msgs/Header:abb_egm_msgs/EGMChannelState"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_egm_msgs
)
_generate_msg_cpp(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_egm_msgs
)

### Generating Services

### Generating Module File
_generate_module_cpp(abb_egm_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_egm_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(abb_egm_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(abb_egm_msgs_generate_messages abb_egm_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_cpp _abb_egm_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_cpp _abb_egm_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_egm_msgs_gencpp)
add_dependencies(abb_egm_msgs_gencpp abb_egm_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_egm_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_egm_msgs
)
_generate_msg_eus(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_egm_msgs
)

### Generating Services

### Generating Module File
_generate_module_eus(abb_egm_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_egm_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(abb_egm_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(abb_egm_msgs_generate_messages abb_egm_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_eus _abb_egm_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_eus _abb_egm_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_egm_msgs_geneus)
add_dependencies(abb_egm_msgs_geneus abb_egm_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_egm_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_egm_msgs
)
_generate_msg_lisp(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_egm_msgs
)

### Generating Services

### Generating Module File
_generate_module_lisp(abb_egm_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_egm_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(abb_egm_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(abb_egm_msgs_generate_messages abb_egm_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_lisp _abb_egm_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_lisp _abb_egm_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_egm_msgs_genlisp)
add_dependencies(abb_egm_msgs_genlisp abb_egm_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_egm_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_egm_msgs
)
_generate_msg_nodejs(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_egm_msgs
)

### Generating Services

### Generating Module File
_generate_module_nodejs(abb_egm_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_egm_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(abb_egm_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(abb_egm_msgs_generate_messages abb_egm_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_nodejs _abb_egm_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_nodejs _abb_egm_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_egm_msgs_gennodejs)
add_dependencies(abb_egm_msgs_gennodejs abb_egm_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_egm_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_egm_msgs
)
_generate_msg_py(abb_egm_msgs
  "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg"
  "${MSG_I_FLAGS}"
  "/home/xi/miniforge3/envs/yumiegmros/share/std_msgs/cmake/../msg/Header.msg;/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_egm_msgs
)

### Generating Services

### Generating Module File
_generate_module_py(abb_egm_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_egm_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(abb_egm_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(abb_egm_msgs_generate_messages abb_egm_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMChannelState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_py _abb_egm_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_egm_msgs/msg/EGMState.msg" NAME_WE)
add_dependencies(abb_egm_msgs_generate_messages_py _abb_egm_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(abb_egm_msgs_genpy)
add_dependencies(abb_egm_msgs_genpy abb_egm_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS abb_egm_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_egm_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/abb_egm_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(abb_egm_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_egm_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/abb_egm_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(abb_egm_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_egm_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/abb_egm_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(abb_egm_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_egm_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/abb_egm_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(abb_egm_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_egm_msgs)
  install(CODE "execute_process(COMMAND \"/home/xi/miniforge3/envs/yumiegmros/bin/python3.11\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_egm_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/abb_egm_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(abb_egm_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
