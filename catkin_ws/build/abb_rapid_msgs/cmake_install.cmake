# Install script for directory: /home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/xi/yumi_ros_noetic/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/xi/miniforge3/envs/yumiegmros/bin/x86_64-conda-linux-gnu-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xi/yumi_ros_noetic/catkin_ws/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/xi/yumi_ros_noetic/catkin_ws/install" TYPE PROGRAM FILES "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/_setup_util.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xi/yumi_ros_noetic/catkin_ws/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/xi/yumi_ros_noetic/catkin_ws/install" TYPE PROGRAM FILES "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/env.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xi/yumi_ros_noetic/catkin_ws/install/setup.bash;/home/xi/yumi_ros_noetic/catkin_ws/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/xi/yumi_ros_noetic/catkin_ws/install" TYPE FILE FILES
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/setup.bash"
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/local_setup.bash"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xi/yumi_ros_noetic/catkin_ws/install/setup.sh;/home/xi/yumi_ros_noetic/catkin_ws/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/xi/yumi_ros_noetic/catkin_ws/install" TYPE FILE FILES
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/setup.sh"
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/local_setup.sh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xi/yumi_ros_noetic/catkin_ws/install/setup.zsh;/home/xi/yumi_ros_noetic/catkin_ws/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/xi/yumi_ros_noetic/catkin_ws/install" TYPE FILE FILES
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/setup.zsh"
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xi/yumi_ros_noetic/catkin_ws/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/xi/yumi_ros_noetic/catkin_ws/install" TYPE FILE FILES "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/.rosinstall")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/abb_rapid_msgs/msg" TYPE FILE FILES
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ConfData.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ExtJoint.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/JointTarget.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/LoadData.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Orient.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pos.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/Pose.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/RobJoint.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/RobTarget.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/ToolData.msg"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/msg/WObjData.msg"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/abb_rapid_msgs/cmake" TYPE FILE FILES "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/abb_rapid_msgs-msg-paths.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/xi/yumi_ros_noetic/catkin_ws/devel/.private/abb_rapid_msgs/include/abb_rapid_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/xi/yumi_ros_noetic/catkin_ws/devel/.private/abb_rapid_msgs/share/roseus/ros/abb_rapid_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/xi/yumi_ros_noetic/catkin_ws/devel/.private/abb_rapid_msgs/share/common-lisp/ros/abb_rapid_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/xi/yumi_ros_noetic/catkin_ws/devel/.private/abb_rapid_msgs/share/gennodejs/ros/abb_rapid_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/xi/miniforge3/envs/yumiegmros/bin/python3.11" -m compileall "/home/xi/yumi_ros_noetic/catkin_ws/devel/.private/abb_rapid_msgs/lib/python3.11/site-packages/abb_rapid_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.11/site-packages" TYPE DIRECTORY FILES "/home/xi/yumi_ros_noetic/catkin_ws/devel/.private/abb_rapid_msgs/lib/python3.11/site-packages/abb_rapid_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/abb_rapid_msgs.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/abb_rapid_msgs/cmake" TYPE FILE FILES "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/abb_rapid_msgs-msg-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/abb_rapid_msgs/cmake" TYPE FILE FILES
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/abb_rapid_msgsConfig.cmake"
    "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/catkin_generated/installspace/abb_rapid_msgsConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/abb_rapid_msgs" TYPE FILE FILES "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/abb_rapid_msgs" TYPE FILE FILES
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/LICENSE"
    "/home/xi/yumi_ros_noetic/catkin_ws/src/abb_robot_driver_interfaces/abb_rapid_msgs/README.md"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/gtest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/xi/yumi_ros_noetic/catkin_ws/build/abb_rapid_msgs/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
