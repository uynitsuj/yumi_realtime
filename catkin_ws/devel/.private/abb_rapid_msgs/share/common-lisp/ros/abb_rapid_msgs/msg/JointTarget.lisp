; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude JointTarget.msg.html

(cl:defclass <JointTarget> (roslisp-msg-protocol:ros-message)
  ((robax
    :reader robax
    :initarg :robax
    :type abb_rapid_msgs-msg:RobJoint
    :initform (cl:make-instance 'abb_rapid_msgs-msg:RobJoint))
   (extax
    :reader extax
    :initarg :extax
    :type abb_rapid_msgs-msg:ExtJoint
    :initform (cl:make-instance 'abb_rapid_msgs-msg:ExtJoint)))
)

(cl:defclass JointTarget (<JointTarget>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <JointTarget>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'JointTarget)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<JointTarget> is deprecated: use abb_rapid_msgs-msg:JointTarget instead.")))

(cl:ensure-generic-function 'robax-val :lambda-list '(m))
(cl:defmethod robax-val ((m <JointTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:robax-val is deprecated.  Use abb_rapid_msgs-msg:robax instead.")
  (robax m))

(cl:ensure-generic-function 'extax-val :lambda-list '(m))
(cl:defmethod extax-val ((m <JointTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:extax-val is deprecated.  Use abb_rapid_msgs-msg:extax instead.")
  (extax m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <JointTarget>) ostream)
  "Serializes a message object of type '<JointTarget>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'robax) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'extax) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <JointTarget>) istream)
  "Deserializes a message object of type '<JointTarget>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'robax) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'extax) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<JointTarget>)))
  "Returns string type for a message object of type '<JointTarget>"
  "abb_rapid_msgs/JointTarget")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'JointTarget)))
  "Returns string type for a message object of type 'JointTarget"
  "abb_rapid_msgs/JointTarget")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<JointTarget>)))
  "Returns md5sum for a message object of type '<JointTarget>"
  "a80f8974c2ebed9e2ef462fcc2a28a38")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'JointTarget)))
  "Returns md5sum for a message object of type 'JointTarget"
  "a80f8974c2ebed9e2ef462fcc2a28a38")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<JointTarget>)))
  "Returns full string definition for message of type '<JointTarget>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"jointtarget\" (joint position data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Axis positions of the robot axes in degrees.~%RobJoint robax~%~%# The position of the external axes.~%ExtJoint extax~%~%================================================================================~%MSG: abb_rapid_msgs/RobJoint~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"robjoint\" (joint position of robot axes).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of robot axis 1-6 in degrees from the calibration position.~%float32 rax_1~%float32 rax_2~%float32 rax_3~%float32 rax_4~%float32 rax_5~%float32 rax_6~%~%================================================================================~%MSG: abb_rapid_msgs/ExtJoint~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"extjoint\" (position of external joints).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of the external logical axis \"a\"-\"f\",~%# expressed in degrees or mm (depending on the type of axis).~%float32 eax_a~%float32 eax_b~%float32 eax_c~%float32 eax_d~%float32 eax_e~%float32 eax_f~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'JointTarget)))
  "Returns full string definition for message of type 'JointTarget"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"jointtarget\" (joint position data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Axis positions of the robot axes in degrees.~%RobJoint robax~%~%# The position of the external axes.~%ExtJoint extax~%~%================================================================================~%MSG: abb_rapid_msgs/RobJoint~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"robjoint\" (joint position of robot axes).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of robot axis 1-6 in degrees from the calibration position.~%float32 rax_1~%float32 rax_2~%float32 rax_3~%float32 rax_4~%float32 rax_5~%float32 rax_6~%~%================================================================================~%MSG: abb_rapid_msgs/ExtJoint~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"extjoint\" (position of external joints).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of the external logical axis \"a\"-\"f\",~%# expressed in degrees or mm (depending on the type of axis).~%float32 eax_a~%float32 eax_b~%float32 eax_c~%float32 eax_d~%float32 eax_e~%float32 eax_f~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <JointTarget>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'robax))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'extax))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <JointTarget>))
  "Converts a ROS message object to a list"
  (cl:list 'JointTarget
    (cl:cons ':robax (robax msg))
    (cl:cons ':extax (extax msg))
))
