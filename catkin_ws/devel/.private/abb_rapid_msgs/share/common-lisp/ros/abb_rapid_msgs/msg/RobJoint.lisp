; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude RobJoint.msg.html

(cl:defclass <RobJoint> (roslisp-msg-protocol:ros-message)
  ((rax_1
    :reader rax_1
    :initarg :rax_1
    :type cl:float
    :initform 0.0)
   (rax_2
    :reader rax_2
    :initarg :rax_2
    :type cl:float
    :initform 0.0)
   (rax_3
    :reader rax_3
    :initarg :rax_3
    :type cl:float
    :initform 0.0)
   (rax_4
    :reader rax_4
    :initarg :rax_4
    :type cl:float
    :initform 0.0)
   (rax_5
    :reader rax_5
    :initarg :rax_5
    :type cl:float
    :initform 0.0)
   (rax_6
    :reader rax_6
    :initarg :rax_6
    :type cl:float
    :initform 0.0))
)

(cl:defclass RobJoint (<RobJoint>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RobJoint>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RobJoint)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<RobJoint> is deprecated: use abb_rapid_msgs-msg:RobJoint instead.")))

(cl:ensure-generic-function 'rax_1-val :lambda-list '(m))
(cl:defmethod rax_1-val ((m <RobJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rax_1-val is deprecated.  Use abb_rapid_msgs-msg:rax_1 instead.")
  (rax_1 m))

(cl:ensure-generic-function 'rax_2-val :lambda-list '(m))
(cl:defmethod rax_2-val ((m <RobJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rax_2-val is deprecated.  Use abb_rapid_msgs-msg:rax_2 instead.")
  (rax_2 m))

(cl:ensure-generic-function 'rax_3-val :lambda-list '(m))
(cl:defmethod rax_3-val ((m <RobJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rax_3-val is deprecated.  Use abb_rapid_msgs-msg:rax_3 instead.")
  (rax_3 m))

(cl:ensure-generic-function 'rax_4-val :lambda-list '(m))
(cl:defmethod rax_4-val ((m <RobJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rax_4-val is deprecated.  Use abb_rapid_msgs-msg:rax_4 instead.")
  (rax_4 m))

(cl:ensure-generic-function 'rax_5-val :lambda-list '(m))
(cl:defmethod rax_5-val ((m <RobJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rax_5-val is deprecated.  Use abb_rapid_msgs-msg:rax_5 instead.")
  (rax_5 m))

(cl:ensure-generic-function 'rax_6-val :lambda-list '(m))
(cl:defmethod rax_6-val ((m <RobJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:rax_6-val is deprecated.  Use abb_rapid_msgs-msg:rax_6 instead.")
  (rax_6 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RobJoint>) ostream)
  "Serializes a message object of type '<RobJoint>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rax_1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rax_2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rax_3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rax_4))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rax_5))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rax_6))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RobJoint>) istream)
  "Deserializes a message object of type '<RobJoint>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rax_1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rax_2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rax_3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rax_4) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rax_5) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rax_6) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RobJoint>)))
  "Returns string type for a message object of type '<RobJoint>"
  "abb_rapid_msgs/RobJoint")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RobJoint)))
  "Returns string type for a message object of type 'RobJoint"
  "abb_rapid_msgs/RobJoint")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RobJoint>)))
  "Returns md5sum for a message object of type '<RobJoint>"
  "24a88117c4b385974e50d44f6663eb2a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RobJoint)))
  "Returns md5sum for a message object of type 'RobJoint"
  "24a88117c4b385974e50d44f6663eb2a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RobJoint>)))
  "Returns full string definition for message of type '<RobJoint>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"robjoint\" (joint position of robot axes).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of robot axis 1-6 in degrees from the calibration position.~%float32 rax_1~%float32 rax_2~%float32 rax_3~%float32 rax_4~%float32 rax_5~%float32 rax_6~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RobJoint)))
  "Returns full string definition for message of type 'RobJoint"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"robjoint\" (joint position of robot axes).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of robot axis 1-6 in degrees from the calibration position.~%float32 rax_1~%float32 rax_2~%float32 rax_3~%float32 rax_4~%float32 rax_5~%float32 rax_6~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RobJoint>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RobJoint>))
  "Converts a ROS message object to a list"
  (cl:list 'RobJoint
    (cl:cons ':rax_1 (rax_1 msg))
    (cl:cons ':rax_2 (rax_2 msg))
    (cl:cons ':rax_3 (rax_3 msg))
    (cl:cons ':rax_4 (rax_4 msg))
    (cl:cons ':rax_5 (rax_5 msg))
    (cl:cons ':rax_6 (rax_6 msg))
))
