; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-msg)


;//! \htmlinclude EGMRunSettings.msg.html

(cl:defclass <EGMRunSettings> (roslisp-msg-protocol:ros-message)
  ((cond_time
    :reader cond_time
    :initarg :cond_time
    :type cl:float
    :initform 0.0)
   (ramp_in_time
    :reader ramp_in_time
    :initarg :ramp_in_time
    :type cl:float
    :initform 0.0)
   (offset
    :reader offset
    :initarg :offset
    :type abb_rapid_msgs-msg:Pose
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pose))
   (pos_corr_gain
    :reader pos_corr_gain
    :initarg :pos_corr_gain
    :type cl:float
    :initform 0.0))
)

(cl:defclass EGMRunSettings (<EGMRunSettings>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EGMRunSettings>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EGMRunSettings)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-msg:<EGMRunSettings> is deprecated: use abb_rapid_sm_addin_msgs-msg:EGMRunSettings instead.")))

(cl:ensure-generic-function 'cond_time-val :lambda-list '(m))
(cl:defmethod cond_time-val ((m <EGMRunSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:cond_time-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:cond_time instead.")
  (cond_time m))

(cl:ensure-generic-function 'ramp_in_time-val :lambda-list '(m))
(cl:defmethod ramp_in_time-val ((m <EGMRunSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:ramp_in_time-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:ramp_in_time instead.")
  (ramp_in_time m))

(cl:ensure-generic-function 'offset-val :lambda-list '(m))
(cl:defmethod offset-val ((m <EGMRunSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:offset-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:offset instead.")
  (offset m))

(cl:ensure-generic-function 'pos_corr_gain-val :lambda-list '(m))
(cl:defmethod pos_corr_gain-val ((m <EGMRunSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:pos_corr_gain-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:pos_corr_gain instead.")
  (pos_corr_gain m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EGMRunSettings>) ostream)
  "Serializes a message object of type '<EGMRunSettings>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'cond_time))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ramp_in_time))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'offset) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'pos_corr_gain))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EGMRunSettings>) istream)
  "Deserializes a message object of type '<EGMRunSettings>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'cond_time) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ramp_in_time) (roslisp-utils:decode-single-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'offset) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'pos_corr_gain) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EGMRunSettings>)))
  "Returns string type for a message object of type '<EGMRunSettings>"
  "abb_rapid_sm_addin_msgs/EGMRunSettings")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EGMRunSettings)))
  "Returns string type for a message object of type 'EGMRunSettings"
  "abb_rapid_sm_addin_msgs/EGMRunSettings")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EGMRunSettings>)))
  "Returns md5sum for a message object of type '<EGMRunSettings>"
  "f6a2d3b6d7d6ffc880c4bc86612a6691")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EGMRunSettings)))
  "Returns md5sum for a message object of type 'EGMRunSettings"
  "f6a2d3b6d7d6ffc880c4bc86612a6691")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EGMRunSettings>)))
  "Returns full string definition for message of type '<EGMRunSettings>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMRunSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMRunSettings\" are for example used with the RAPID EGM instructions~%#   \"EGMRunJoint\" and \"EGMRunPose\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Condition time [s] for how long an EGM motion's convergence criteria must be~%# fulfilled before the motion is considered to be finished.~%float32 cond_time~%~%# Ramp in time [s].~%float32 ramp_in_time~%~%# Static offset applied on top of the EGM references~%# (only used in EGM pose mode).~%abb_rapid_msgs/Pose offset~%~%# Position correction gain for the EGM controller.~%float32 pos_corr_gain~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EGMRunSettings)))
  "Returns full string definition for message of type 'EGMRunSettings"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMRunSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMRunSettings\" are for example used with the RAPID EGM instructions~%#   \"EGMRunJoint\" and \"EGMRunPose\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Condition time [s] for how long an EGM motion's convergence criteria must be~%# fulfilled before the motion is considered to be finished.~%float32 cond_time~%~%# Ramp in time [s].~%float32 ramp_in_time~%~%# Static offset applied on top of the EGM references~%# (only used in EGM pose mode).~%abb_rapid_msgs/Pose offset~%~%# Position correction gain for the EGM controller.~%float32 pos_corr_gain~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EGMRunSettings>))
  (cl:+ 0
     4
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'offset))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EGMRunSettings>))
  "Converts a ROS message object to a list"
  (cl:list 'EGMRunSettings
    (cl:cons ':cond_time (cond_time msg))
    (cl:cons ':ramp_in_time (ramp_in_time msg))
    (cl:cons ':offset (offset msg))
    (cl:cons ':pos_corr_gain (pos_corr_gain msg))
))
