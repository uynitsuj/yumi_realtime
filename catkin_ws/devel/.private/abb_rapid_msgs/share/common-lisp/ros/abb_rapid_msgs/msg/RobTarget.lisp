; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude RobTarget.msg.html

(cl:defclass <RobTarget> (roslisp-msg-protocol:ros-message)
  ((trans
    :reader trans
    :initarg :trans
    :type abb_rapid_msgs-msg:Pos
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pos))
   (rot
    :reader rot
    :initarg :rot
    :type abb_rapid_msgs-msg:Orient
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Orient))
   (robconf
    :reader robconf
    :initarg :robconf
    :type abb_rapid_msgs-msg:ConfData
    :initform (cl:make-instance 'abb_rapid_msgs-msg:ConfData))
   (extax
    :reader extax
    :initarg :extax
    :type abb_rapid_msgs-msg:ExtJoint
    :initform (cl:make-instance 'abb_rapid_msgs-msg:ExtJoint)))
)

(cl:defclass RobTarget (<RobTarget>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RobTarget>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RobTarget)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<RobTarget> is deprecated: use abb_rapid_msgs-msg:RobTarget instead.")))

(cl:ensure-generic-function 'trans-val :lambda-list '(m))
(cl:defmethod trans-val ((m <RobTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:trans-val is deprecated.  Use abb_rapid_msgs-msg:trans instead.")
  (trans m))

(cl:ensure-generic-function 'rot-val :lambda-list '(m))
(cl:defmethod rot-val ((m <RobTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rot-val is deprecated.  Use abb_rapid_msgs-msg:rot instead.")
  (rot m))

(cl:ensure-generic-function 'robconf-val :lambda-list '(m))
(cl:defmethod robconf-val ((m <RobTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:robconf-val is deprecated.  Use abb_rapid_msgs-msg:robconf instead.")
  (robconf m))

(cl:ensure-generic-function 'extax-val :lambda-list '(m))
(cl:defmethod extax-val ((m <RobTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:extax-val is deprecated.  Use abb_rapid_msgs-msg:extax instead.")
  (extax m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RobTarget>) ostream)
  "Serializes a message object of type '<RobTarget>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'trans) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rot) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'robconf) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'extax) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RobTarget>) istream)
  "Deserializes a message object of type '<RobTarget>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'trans) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rot) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'robconf) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'extax) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RobTarget>)))
  "Returns string type for a message object of type '<RobTarget>"
  "abb_rapid_msgs/RobTarget")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RobTarget)))
  "Returns string type for a message object of type 'RobTarget"
  "abb_rapid_msgs/RobTarget")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RobTarget>)))
  "Returns md5sum for a message object of type '<RobTarget>"
  "31d5c046c0fae9eadd8133382f15db0a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RobTarget)))
  "Returns md5sum for a message object of type 'RobTarget"
  "31d5c046c0fae9eadd8133382f15db0a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RobTarget>)))
  "Returns full string definition for message of type '<RobTarget>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"robtarget\" (position data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position (x, y, and z) of the tool center point expressed in mm.~%#~%# The position is specified in relation to the current object coordinate system,~%# including program displacement.~%#~%# If no work object is specified then this is the world coordinate system.~%Pos trans~%~%# The orientation of the tool, expressed in the~%# form of a quaternion (q1, q2, q3, and q4).~%#~%# The orientation is specified in relation to the current~%# object coordinate system including program displacement.~%#~%# If no work object is specified then this is the world coordinate system.~%Orient rot~%~%# The axis configuration of the robot (cf1, cf4, cf6, and cfx).~%#~%# This is defined in the form of the current quarter~%# revolution of axis 1, axis 4, and axis 6.~%#~%# The first positive quarter revolution 0 to 90 degrees is defined as 0.~%#~%# The meaning of the component cfx is dependent on robot type.~%ConfData robconf~%~%# The position of the additional axes.~%ExtJoint extax~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%================================================================================~%MSG: abb_rapid_msgs/ConfData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"confdata\" (robot configuration data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Rotating axis:~%# - The current quadrant of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%float32 cf1~%float32 cf4~%float32 cf6~%~%# Rotating axis:~%# - For serial link robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For SCARA robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 1.~%# - For 7-axis robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For paint robots, the current quadrant of axis 5,~%#   expressed as a positive or negative integer.~%# - For other robots, using the current quadrant of axis 2,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 2,~%#   expressed as a positive or negative integer.~%float32 cfx~%~%================================================================================~%MSG: abb_rapid_msgs/ExtJoint~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"extjoint\" (position of external joints).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of the external logical axis \"a\"-\"f\",~%# expressed in degrees or mm (depending on the type of axis).~%float32 eax_a~%float32 eax_b~%float32 eax_c~%float32 eax_d~%float32 eax_e~%float32 eax_f~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RobTarget)))
  "Returns full string definition for message of type 'RobTarget"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"robtarget\" (position data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position (x, y, and z) of the tool center point expressed in mm.~%#~%# The position is specified in relation to the current object coordinate system,~%# including program displacement.~%#~%# If no work object is specified then this is the world coordinate system.~%Pos trans~%~%# The orientation of the tool, expressed in the~%# form of a quaternion (q1, q2, q3, and q4).~%#~%# The orientation is specified in relation to the current~%# object coordinate system including program displacement.~%#~%# If no work object is specified then this is the world coordinate system.~%Orient rot~%~%# The axis configuration of the robot (cf1, cf4, cf6, and cfx).~%#~%# This is defined in the form of the current quarter~%# revolution of axis 1, axis 4, and axis 6.~%#~%# The first positive quarter revolution 0 to 90 degrees is defined as 0.~%#~%# The meaning of the component cfx is dependent on robot type.~%ConfData robconf~%~%# The position of the additional axes.~%ExtJoint extax~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%================================================================================~%MSG: abb_rapid_msgs/ConfData~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"confdata\" (robot configuration data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Rotating axis:~%# - The current quadrant of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%float32 cf1~%float32 cf4~%float32 cf6~%~%# Rotating axis:~%# - For serial link robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For SCARA robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 1.~%# - For 7-axis robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For paint robots, the current quadrant of axis 5,~%#   expressed as a positive or negative integer.~%# - For other robots, using the current quadrant of axis 2,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 2,~%#   expressed as a positive or negative integer.~%float32 cfx~%~%================================================================================~%MSG: abb_rapid_msgs/ExtJoint~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"extjoint\" (position of external joints).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of the external logical axis \"a\"-\"f\",~%# expressed in degrees or mm (depending on the type of axis).~%float32 eax_a~%float32 eax_b~%float32 eax_c~%float32 eax_d~%float32 eax_e~%float32 eax_f~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RobTarget>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'trans))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rot))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'robconf))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'extax))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RobTarget>))
  "Converts a ROS message object to a list"
  (cl:list 'RobTarget
    (cl:cons ':trans (trans msg))
    (cl:cons ':rot (rot msg))
    (cl:cons ':robconf (robconf msg))
    (cl:cons ':extax (extax msg))
))
