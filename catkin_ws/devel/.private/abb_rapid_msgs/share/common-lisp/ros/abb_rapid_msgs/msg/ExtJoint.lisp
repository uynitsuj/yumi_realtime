; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude ExtJoint.msg.html

(cl:defclass <ExtJoint> (roslisp-msg-protocol:ros-message)
  ((eax_a
    :reader eax_a
    :initarg :eax_a
    :type cl:float
    :initform 0.0)
   (eax_b
    :reader eax_b
    :initarg :eax_b
    :type cl:float
    :initform 0.0)
   (eax_c
    :reader eax_c
    :initarg :eax_c
    :type cl:float
    :initform 0.0)
   (eax_d
    :reader eax_d
    :initarg :eax_d
    :type cl:float
    :initform 0.0)
   (eax_e
    :reader eax_e
    :initarg :eax_e
    :type cl:float
    :initform 0.0)
   (eax_f
    :reader eax_f
    :initarg :eax_f
    :type cl:float
    :initform 0.0))
)

(cl:defclass ExtJoint (<ExtJoint>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ExtJoint>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ExtJoint)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<ExtJoint> is deprecated: use abb_rapid_msgs-msg:ExtJoint instead.")))

(cl:ensure-generic-function 'eax_a-val :lambda-list '(m))
(cl:defmethod eax_a-val ((m <ExtJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:eax_a-val is deprecated.  Use abb_rapid_msgs-msg:eax_a instead.")
  (eax_a m))

(cl:ensure-generic-function 'eax_b-val :lambda-list '(m))
(cl:defmethod eax_b-val ((m <ExtJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:eax_b-val is deprecated.  Use abb_rapid_msgs-msg:eax_b instead.")
  (eax_b m))

(cl:ensure-generic-function 'eax_c-val :lambda-list '(m))
(cl:defmethod eax_c-val ((m <ExtJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:eax_c-val is deprecated.  Use abb_rapid_msgs-msg:eax_c instead.")
  (eax_c m))

(cl:ensure-generic-function 'eax_d-val :lambda-list '(m))
(cl:defmethod eax_d-val ((m <ExtJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:eax_d-val is deprecated.  Use abb_rapid_msgs-msg:eax_d instead.")
  (eax_d m))

(cl:ensure-generic-function 'eax_e-val :lambda-list '(m))
(cl:defmethod eax_e-val ((m <ExtJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:eax_e-val is deprecated.  Use abb_rapid_msgs-msg:eax_e instead.")
  (eax_e m))

(cl:ensure-generic-function 'eax_f-val :lambda-list '(m))
(cl:defmethod eax_f-val ((m <ExtJoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:eax_f-val is deprecated.  Use abb_rapid_msgs-msg:eax_f instead.")
  (eax_f m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ExtJoint>) ostream)
  "Serializes a message object of type '<ExtJoint>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'eax_a))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'eax_b))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'eax_c))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'eax_d))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'eax_e))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'eax_f))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ExtJoint>) istream)
  "Deserializes a message object of type '<ExtJoint>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'eax_a) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'eax_b) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'eax_c) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'eax_d) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'eax_e) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'eax_f) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ExtJoint>)))
  "Returns string type for a message object of type '<ExtJoint>"
  "abb_rapid_msgs/ExtJoint")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ExtJoint)))
  "Returns string type for a message object of type 'ExtJoint"
  "abb_rapid_msgs/ExtJoint")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ExtJoint>)))
  "Returns md5sum for a message object of type '<ExtJoint>"
  "f8550d9e82bcfae84c73de38fde07b03")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ExtJoint)))
  "Returns md5sum for a message object of type 'ExtJoint"
  "f8550d9e82bcfae84c73de38fde07b03")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ExtJoint>)))
  "Returns full string definition for message of type '<ExtJoint>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"extjoint\" (position of external joints).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of the external logical axis \"a\"-\"f\",~%# expressed in degrees or mm (depending on the type of axis).~%float32 eax_a~%float32 eax_b~%float32 eax_c~%float32 eax_d~%float32 eax_e~%float32 eax_f~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ExtJoint)))
  "Returns full string definition for message of type 'ExtJoint"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"extjoint\" (position of external joints).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The position of the external logical axis \"a\"-\"f\",~%# expressed in degrees or mm (depending on the type of axis).~%float32 eax_a~%float32 eax_b~%float32 eax_c~%float32 eax_d~%float32 eax_e~%float32 eax_f~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ExtJoint>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ExtJoint>))
  "Converts a ROS message object to a list"
  (cl:list 'ExtJoint
    (cl:cons ':eax_a (eax_a msg))
    (cl:cons ':eax_b (eax_b msg))
    (cl:cons ':eax_c (eax_c msg))
    (cl:cons ':eax_d (eax_d msg))
    (cl:cons ':eax_e (eax_e msg))
    (cl:cons ':eax_f (eax_f msg))
))
