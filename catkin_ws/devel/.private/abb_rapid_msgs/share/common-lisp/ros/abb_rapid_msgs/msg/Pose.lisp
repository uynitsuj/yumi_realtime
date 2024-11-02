; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude Pose.msg.html

(cl:defclass <Pose> (roslisp-msg-protocol:ros-message)
  ((trans
    :reader trans
    :initarg :trans
    :type abb_rapid_msgs-msg:Pos
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pos))
   (rot
    :reader rot
    :initarg :rot
    :type abb_rapid_msgs-msg:Orient
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Orient)))
)

(cl:defclass Pose (<Pose>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Pose>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Pose)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<Pose> is deprecated: use abb_rapid_msgs-msg:Pose instead.")))

(cl:ensure-generic-function 'trans-val :lambda-list '(m))
(cl:defmethod trans-val ((m <Pose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:trans-val is deprecated.  Use abb_rapid_msgs-msg:trans instead.")
  (trans m))

(cl:ensure-generic-function 'rot-val :lambda-list '(m))
(cl:defmethod rot-val ((m <Pose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rot-val is deprecated.  Use abb_rapid_msgs-msg:rot instead.")
  (rot m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Pose>) ostream)
  "Serializes a message object of type '<Pose>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'trans) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rot) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Pose>) istream)
  "Deserializes a message object of type '<Pose>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'trans) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rot) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Pose>)))
  "Returns string type for a message object of type '<Pose>"
  "abb_rapid_msgs/Pose")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Pose)))
  "Returns string type for a message object of type 'Pose"
  "abb_rapid_msgs/Pose")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Pose>)))
  "Returns md5sum for a message object of type '<Pose>"
  "94c76c2f70e3c86d018090698fefbfdc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Pose)))
  "Returns md5sum for a message object of type 'Pose"
  "94c76c2f70e3c86d018090698fefbfdc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Pose>)))
  "Returns full string definition for message of type '<Pose>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Pose)))
  "Returns full string definition for message of type 'Pose"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Pose>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'trans))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rot))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Pose>))
  "Converts a ROS message object to a list"
  (cl:list 'Pose
    (cl:cons ':trans (trans msg))
    (cl:cons ':rot (rot msg))
))
