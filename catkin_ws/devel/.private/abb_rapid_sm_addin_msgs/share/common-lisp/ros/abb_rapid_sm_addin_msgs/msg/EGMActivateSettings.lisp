; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-msg)


;//! \htmlinclude EGMActivateSettings.msg.html

(cl:defclass <EGMActivateSettings> (roslisp-msg-protocol:ros-message)
  ((tool
    :reader tool
    :initarg :tool
    :type abb_rapid_msgs-msg:ToolData
    :initform (cl:make-instance 'abb_rapid_msgs-msg:ToolData))
   (wobj
    :reader wobj
    :initarg :wobj
    :type abb_rapid_msgs-msg:WObjData
    :initform (cl:make-instance 'abb_rapid_msgs-msg:WObjData))
   (correction_frame
    :reader correction_frame
    :initarg :correction_frame
    :type abb_rapid_msgs-msg:Pose
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pose))
   (sensor_frame
    :reader sensor_frame
    :initarg :sensor_frame
    :type abb_rapid_msgs-msg:Pose
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pose))
   (cond_min_max
    :reader cond_min_max
    :initarg :cond_min_max
    :type cl:float
    :initform 0.0)
   (lp_filter
    :reader lp_filter
    :initarg :lp_filter
    :type cl:float
    :initform 0.0)
   (sample_rate
    :reader sample_rate
    :initarg :sample_rate
    :type cl:fixnum
    :initform 0)
   (max_speed_deviation
    :reader max_speed_deviation
    :initarg :max_speed_deviation
    :type cl:float
    :initform 0.0))
)

(cl:defclass EGMActivateSettings (<EGMActivateSettings>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EGMActivateSettings>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EGMActivateSettings)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-msg:<EGMActivateSettings> is deprecated: use abb_rapid_sm_addin_msgs-msg:EGMActivateSettings instead.")))

(cl:ensure-generic-function 'tool-val :lambda-list '(m))
(cl:defmethod tool-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:tool-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:tool instead.")
  (tool m))

(cl:ensure-generic-function 'wobj-val :lambda-list '(m))
(cl:defmethod wobj-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:wobj-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:wobj instead.")
  (wobj m))

(cl:ensure-generic-function 'correction_frame-val :lambda-list '(m))
(cl:defmethod correction_frame-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:correction_frame-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:correction_frame instead.")
  (correction_frame m))

(cl:ensure-generic-function 'sensor_frame-val :lambda-list '(m))
(cl:defmethod sensor_frame-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:sensor_frame-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:sensor_frame instead.")
  (sensor_frame m))

(cl:ensure-generic-function 'cond_min_max-val :lambda-list '(m))
(cl:defmethod cond_min_max-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:cond_min_max-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:cond_min_max instead.")
  (cond_min_max m))

(cl:ensure-generic-function 'lp_filter-val :lambda-list '(m))
(cl:defmethod lp_filter-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:lp_filter-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:lp_filter instead.")
  (lp_filter m))

(cl:ensure-generic-function 'sample_rate-val :lambda-list '(m))
(cl:defmethod sample_rate-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:sample_rate-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:sample_rate instead.")
  (sample_rate m))

(cl:ensure-generic-function 'max_speed_deviation-val :lambda-list '(m))
(cl:defmethod max_speed_deviation-val ((m <EGMActivateSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:max_speed_deviation-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:max_speed_deviation instead.")
  (max_speed_deviation m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EGMActivateSettings>) ostream)
  "Serializes a message object of type '<EGMActivateSettings>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tool) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'wobj) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'correction_frame) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'sensor_frame) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'cond_min_max))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lp_filter))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sample_rate)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'max_speed_deviation))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EGMActivateSettings>) istream)
  "Deserializes a message object of type '<EGMActivateSettings>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tool) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'wobj) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'correction_frame) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'sensor_frame) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'cond_min_max) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lp_filter) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sample_rate)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'max_speed_deviation) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EGMActivateSettings>)))
  "Returns string type for a message object of type '<EGMActivateSettings>"
  "abb_rapid_sm_addin_msgs/EGMActivateSettings")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EGMActivateSettings)))
  "Returns string type for a message object of type 'EGMActivateSettings"
  "abb_rapid_sm_addin_msgs/EGMActivateSettings")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EGMActivateSettings>)))
  "Returns md5sum for a message object of type '<EGMActivateSettings>"
  "edc93be8ce076e9147c54b4dc80e7974")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EGMActivateSettings)))
  "Returns md5sum for a message object of type 'EGMActivateSettings"
  "edc93be8ce076e9147c54b4dc80e7974")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EGMActivateSettings>)))
  "Returns full string definition for message of type '<EGMActivateSettings>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMActivateSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMActivateSettings\" are for example used with the RAPID EGM instructions~%#   \"EGMActJoint\" and \"EGMActPose\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Tool to use for EGM motions.~%abb_rapid_msgs/ToolData tool~%~%# Work object to use for EGM motions.~%abb_rapid_msgs/WObjData wobj~%~%# Correction frame to use for EGM motions (only used in EGM pose mode).~%abb_rapid_msgs/Pose correction_frame~%~%# Sensor frame to use for EGM motions (only used in EGM pose mode).~%abb_rapid_msgs/Pose sensor_frame~%~%# Condition threshold value [deg or mm] for when an EGM motion's convergence~%# criteria is considered to be fulfilled.~%float32 cond_min_max~%~%# Low pass filter bandwidth [Hz] for the EGM controller.~%float32 lp_filter~%~%# Sample rate [ms] for the EGM communication.~%uint8 sample_rate~%~%# Maximum admitted joint speed [deg/s].~%float32 max_speed_deviation~%~%================================================================================~%MSG: abb_rapid_msgs/ToolData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"tooldata\" (tool data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot is holding the tool:~%# - TRUE: The robot is holding the tool.~%# - FALSE: The robot is not holding the tool, that is, a stationary tool.~%bool robhold~%~%# The tool coordinate system, that is:~%# - The position of the TCP (x, y and z) in mm,~%#   expressed in the wrist coordinate system (tool0).~%# - The orientation of the tool coordinate system,~%#   expressed in the wrist coordinate system.~%Pose tframe~%~%# The load of the tool (if robot held tool).~%# The load of the gripper holding the work object (if stationary tool).~%LoadData tload~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%================================================================================~%MSG: abb_rapid_msgs/LoadData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"loaddata\" (load data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mass (weight) of the load in kg.~%float32 mass~%~%# The center of gravity of the payload, expressed in mm,~%# in the tool coordinate system if the robot is holding the tool.~%#~%# If a stationary tool is used then the center of gravity for the payload~%# held by the gripper is expressed in the object frame of the work object~%# coordinate system moved by the robot.~%Pos cog~%~%# The orientation of the axes of moment.~%#~%# These are the principal axes of the payload~%# moment of inertia with origin in cog.~%#~%# If the robot is holding the tool, the axes of moment~%# are expressed in the tool coordinate system.~%Orient aom~%~%# The moment of inertia of the load around the x-, y- and z-axis,~%# expressed in kgm^2.~%float32 ix~%float32 iy~%float32 iz~%~%================================================================================~%MSG: abb_rapid_msgs/WObjData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"wobjdata\" (work object data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot in the actual~%# program task is holding the work object:~%# - TRUE: The robot is holding the work object,~%#         i.e. using a stationary tool.~%# - FALSE: The robot is not holding the work object,~%#          i.e. the robot is holding the tool.~%bool robhold~%~%# Defines whether or not a fixed user coordinate system is used:~%# - TRUE: Fixed user coordinate system.~%# - FALSE: Movable user coordinate system,~%#          i.e. coordinated external axes are used.~%#          Also to be used in a MultiMove system in~%#          semicoordinated or synchronized coordinated mode.~%bool ufprog~%~%# The mechanical unit with which the robot movements are coordinated.~%#~%# Only specified in the case of movable user~%# coordinate systems (ufprog is FALSE).~%string ufmec~%~%# The user coordinate system, i.e. the position~%# of the current work surface or fixture.~%Pose uframe~%~%# The object coordinate system, i.e. the position~%# of the current work object.~%Pose oframe~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EGMActivateSettings)))
  "Returns full string definition for message of type 'EGMActivateSettings"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMActivateSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMActivateSettings\" are for example used with the RAPID EGM instructions~%#   \"EGMActJoint\" and \"EGMActPose\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Tool to use for EGM motions.~%abb_rapid_msgs/ToolData tool~%~%# Work object to use for EGM motions.~%abb_rapid_msgs/WObjData wobj~%~%# Correction frame to use for EGM motions (only used in EGM pose mode).~%abb_rapid_msgs/Pose correction_frame~%~%# Sensor frame to use for EGM motions (only used in EGM pose mode).~%abb_rapid_msgs/Pose sensor_frame~%~%# Condition threshold value [deg or mm] for when an EGM motion's convergence~%# criteria is considered to be fulfilled.~%float32 cond_min_max~%~%# Low pass filter bandwidth [Hz] for the EGM controller.~%float32 lp_filter~%~%# Sample rate [ms] for the EGM communication.~%uint8 sample_rate~%~%# Maximum admitted joint speed [deg/s].~%float32 max_speed_deviation~%~%================================================================================~%MSG: abb_rapid_msgs/ToolData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"tooldata\" (tool data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot is holding the tool:~%# - TRUE: The robot is holding the tool.~%# - FALSE: The robot is not holding the tool, that is, a stationary tool.~%bool robhold~%~%# The tool coordinate system, that is:~%# - The position of the TCP (x, y and z) in mm,~%#   expressed in the wrist coordinate system (tool0).~%# - The orientation of the tool coordinate system,~%#   expressed in the wrist coordinate system.~%Pose tframe~%~%# The load of the tool (if robot held tool).~%# The load of the gripper holding the work object (if stationary tool).~%LoadData tload~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%================================================================================~%MSG: abb_rapid_msgs/LoadData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"loaddata\" (load data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mass (weight) of the load in kg.~%float32 mass~%~%# The center of gravity of the payload, expressed in mm,~%# in the tool coordinate system if the robot is holding the tool.~%#~%# If a stationary tool is used then the center of gravity for the payload~%# held by the gripper is expressed in the object frame of the work object~%# coordinate system moved by the robot.~%Pos cog~%~%# The orientation of the axes of moment.~%#~%# These are the principal axes of the payload~%# moment of inertia with origin in cog.~%#~%# If the robot is holding the tool, the axes of moment~%# are expressed in the tool coordinate system.~%Orient aom~%~%# The moment of inertia of the load around the x-, y- and z-axis,~%# expressed in kgm^2.~%float32 ix~%float32 iy~%float32 iz~%~%================================================================================~%MSG: abb_rapid_msgs/WObjData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"wobjdata\" (work object data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot in the actual~%# program task is holding the work object:~%# - TRUE: The robot is holding the work object,~%#         i.e. using a stationary tool.~%# - FALSE: The robot is not holding the work object,~%#          i.e. the robot is holding the tool.~%bool robhold~%~%# Defines whether or not a fixed user coordinate system is used:~%# - TRUE: Fixed user coordinate system.~%# - FALSE: Movable user coordinate system,~%#          i.e. coordinated external axes are used.~%#          Also to be used in a MultiMove system in~%#          semicoordinated or synchronized coordinated mode.~%bool ufprog~%~%# The mechanical unit with which the robot movements are coordinated.~%#~%# Only specified in the case of movable user~%# coordinate systems (ufprog is FALSE).~%string ufmec~%~%# The user coordinate system, i.e. the position~%# of the current work surface or fixture.~%Pose uframe~%~%# The object coordinate system, i.e. the position~%# of the current work object.~%Pose oframe~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EGMActivateSettings>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tool))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'wobj))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'correction_frame))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'sensor_frame))
     4
     4
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EGMActivateSettings>))
  "Converts a ROS message object to a list"
  (cl:list 'EGMActivateSettings
    (cl:cons ':tool (tool msg))
    (cl:cons ':wobj (wobj msg))
    (cl:cons ':correction_frame (correction_frame msg))
    (cl:cons ':sensor_frame (sensor_frame msg))
    (cl:cons ':cond_min_max (cond_min_max msg))
    (cl:cons ':lp_filter (lp_filter msg))
    (cl:cons ':sample_rate (sample_rate msg))
    (cl:cons ':max_speed_deviation (max_speed_deviation msg))
))
