; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude ToolData.msg.html

(cl:defclass <ToolData> (roslisp-msg-protocol:ros-message)
  ((robhold
    :reader robhold
    :initarg :robhold
    :type cl:boolean
    :initform cl:nil)
   (tframe
    :reader tframe
    :initarg :tframe
    :type abb_rapid_msgs-msg:Pose
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pose))
   (tload
    :reader tload
    :initarg :tload
    :type abb_rapid_msgs-msg:LoadData
    :initform (cl:make-instance 'abb_rapid_msgs-msg:LoadData)))
)

(cl:defclass ToolData (<ToolData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ToolData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ToolData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<ToolData> is deprecated: use abb_rapid_msgs-msg:ToolData instead.")))

(cl:ensure-generic-function 'robhold-val :lambda-list '(m))
(cl:defmethod robhold-val ((m <ToolData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:robhold-val is deprecated.  Use abb_rapid_msgs-msg:robhold instead.")
  (robhold m))

(cl:ensure-generic-function 'tframe-val :lambda-list '(m))
(cl:defmethod tframe-val ((m <ToolData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:tframe-val is deprecated.  Use abb_rapid_msgs-msg:tframe instead.")
  (tframe m))

(cl:ensure-generic-function 'tload-val :lambda-list '(m))
(cl:defmethod tload-val ((m <ToolData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:tload-val is deprecated.  Use abb_rapid_msgs-msg:tload instead.")
  (tload m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ToolData>) ostream)
  "Serializes a message object of type '<ToolData>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'robhold) 1 0)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tframe) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tload) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ToolData>) istream)
  "Deserializes a message object of type '<ToolData>"
    (cl:setf (cl:slot-value msg 'robhold) (cl:not (cl:zerop (cl:read-byte istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tframe) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tload) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ToolData>)))
  "Returns string type for a message object of type '<ToolData>"
  "abb_rapid_msgs/ToolData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ToolData)))
  "Returns string type for a message object of type 'ToolData"
  "abb_rapid_msgs/ToolData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ToolData>)))
  "Returns md5sum for a message object of type '<ToolData>"
  "2d526e4dd17e28147b27f746cf88c5b6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ToolData)))
  "Returns md5sum for a message object of type 'ToolData"
  "2d526e4dd17e28147b27f746cf88c5b6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ToolData>)))
  "Returns full string definition for message of type '<ToolData>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"tooldata\" (tool data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot is holding the tool:~%# - TRUE: The robot is holding the tool.~%# - FALSE: The robot is not holding the tool, that is, a stationary tool.~%bool robhold~%~%# The tool coordinate system, that is:~%# - The position of the TCP (x, y and z) in mm,~%#   expressed in the wrist coordinate system (tool0).~%# - The orientation of the tool coordinate system,~%#   expressed in the wrist coordinate system.~%Pose tframe~%~%# The load of the tool (if robot held tool).~%# The load of the gripper holding the work object (if stationary tool).~%LoadData tload~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%================================================================================~%MSG: abb_rapid_msgs/LoadData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"loaddata\" (load data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mass (weight) of the load in kg.~%float32 mass~%~%# The center of gravity of the payload, expressed in mm,~%# in the tool coordinate system if the robot is holding the tool.~%#~%# If a stationary tool is used then the center of gravity for the payload~%# held by the gripper is expressed in the object frame of the work object~%# coordinate system moved by the robot.~%Pos cog~%~%# The orientation of the axes of moment.~%#~%# These are the principal axes of the payload~%# moment of inertia with origin in cog.~%#~%# If the robot is holding the tool, the axes of moment~%# are expressed in the tool coordinate system.~%Orient aom~%~%# The moment of inertia of the load around the x-, y- and z-axis,~%# expressed in kgm^2.~%float32 ix~%float32 iy~%float32 iz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ToolData)))
  "Returns full string definition for message of type 'ToolData"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"tooldata\" (tool data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot is holding the tool:~%# - TRUE: The robot is holding the tool.~%# - FALSE: The robot is not holding the tool, that is, a stationary tool.~%bool robhold~%~%# The tool coordinate system, that is:~%# - The position of the TCP (x, y and z) in mm,~%#   expressed in the wrist coordinate system (tool0).~%# - The orientation of the tool coordinate system,~%#   expressed in the wrist coordinate system.~%Pose tframe~%~%# The load of the tool (if robot held tool).~%# The load of the gripper holding the work object (if stationary tool).~%LoadData tload~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%================================================================================~%MSG: abb_rapid_msgs/LoadData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"loaddata\" (load data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mass (weight) of the load in kg.~%float32 mass~%~%# The center of gravity of the payload, expressed in mm,~%# in the tool coordinate system if the robot is holding the tool.~%#~%# If a stationary tool is used then the center of gravity for the payload~%# held by the gripper is expressed in the object frame of the work object~%# coordinate system moved by the robot.~%Pos cog~%~%# The orientation of the axes of moment.~%#~%# These are the principal axes of the payload~%# moment of inertia with origin in cog.~%#~%# If the robot is holding the tool, the axes of moment~%# are expressed in the tool coordinate system.~%Orient aom~%~%# The moment of inertia of the load around the x-, y- and z-axis,~%# expressed in kgm^2.~%float32 ix~%float32 iy~%float32 iz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ToolData>))
  (cl:+ 0
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tframe))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tload))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ToolData>))
  "Converts a ROS message object to a list"
  (cl:list 'ToolData
    (cl:cons ':robhold (robhold msg))
    (cl:cons ':tframe (tframe msg))
    (cl:cons ':tload (tload msg))
))
