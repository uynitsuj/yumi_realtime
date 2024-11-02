; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-msg)


;//! \htmlinclude MechanicalUnitState.msg.html

(cl:defclass <MechanicalUnitState> (roslisp-msg-protocol:ros-message)
  ((name
    :reader name
    :initarg :name
    :type cl:string
    :initform "")
   (activated
    :reader activated
    :initarg :activated
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass MechanicalUnitState (<MechanicalUnitState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MechanicalUnitState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MechanicalUnitState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-msg:<MechanicalUnitState> is deprecated: use abb_robot_msgs-msg:MechanicalUnitState instead.")))

(cl:ensure-generic-function 'name-val :lambda-list '(m))
(cl:defmethod name-val ((m <MechanicalUnitState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:name-val is deprecated.  Use abb_robot_msgs-msg:name instead.")
  (name m))

(cl:ensure-generic-function 'activated-val :lambda-list '(m))
(cl:defmethod activated-val ((m <MechanicalUnitState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:activated-val is deprecated.  Use abb_robot_msgs-msg:activated instead.")
  (activated m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MechanicalUnitState>) ostream)
  "Serializes a message object of type '<MechanicalUnitState>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'name))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'activated) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MechanicalUnitState>) istream)
  "Deserializes a message object of type '<MechanicalUnitState>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'activated) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MechanicalUnitState>)))
  "Returns string type for a message object of type '<MechanicalUnitState>"
  "abb_robot_msgs/MechanicalUnitState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MechanicalUnitState)))
  "Returns string type for a message object of type 'MechanicalUnitState"
  "abb_robot_msgs/MechanicalUnitState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MechanicalUnitState>)))
  "Returns md5sum for a message object of type '<MechanicalUnitState>"
  "add093e2c7fdce39ce8a0c1e573668e8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MechanicalUnitState)))
  "Returns md5sum for a message object of type 'MechanicalUnitState"
  "add093e2c7fdce39ce8a0c1e573668e8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MechanicalUnitState>)))
  "Returns full string definition for message of type '<MechanicalUnitState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a mechanical unit defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mechanical unit's name.~%string name~%~%# The mechanical unit's activated/deactivated status.~%bool activated~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MechanicalUnitState)))
  "Returns full string definition for message of type 'MechanicalUnitState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a mechanical unit defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mechanical unit's name.~%string name~%~%# The mechanical unit's activated/deactivated status.~%bool activated~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MechanicalUnitState>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'name))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MechanicalUnitState>))
  "Converts a ROS message object to a list"
  (cl:list 'MechanicalUnitState
    (cl:cons ':name (name msg))
    (cl:cons ':activated (activated msg))
))
