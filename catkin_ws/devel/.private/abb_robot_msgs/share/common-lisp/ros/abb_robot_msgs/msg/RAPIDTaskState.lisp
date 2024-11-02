; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-msg)


;//! \htmlinclude RAPIDTaskState.msg.html

(cl:defclass <RAPIDTaskState> (roslisp-msg-protocol:ros-message)
  ((name
    :reader name
    :initarg :name
    :type cl:string
    :initform "")
   (activated
    :reader activated
    :initarg :activated
    :type cl:boolean
    :initform cl:nil)
   (execution_state
    :reader execution_state
    :initarg :execution_state
    :type cl:fixnum
    :initform 0)
   (motion_task
    :reader motion_task
    :initarg :motion_task
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass RAPIDTaskState (<RAPIDTaskState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RAPIDTaskState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RAPIDTaskState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-msg:<RAPIDTaskState> is deprecated: use abb_robot_msgs-msg:RAPIDTaskState instead.")))

(cl:ensure-generic-function 'name-val :lambda-list '(m))
(cl:defmethod name-val ((m <RAPIDTaskState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:name-val is deprecated.  Use abb_robot_msgs-msg:name instead.")
  (name m))

(cl:ensure-generic-function 'activated-val :lambda-list '(m))
(cl:defmethod activated-val ((m <RAPIDTaskState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:activated-val is deprecated.  Use abb_robot_msgs-msg:activated instead.")
  (activated m))

(cl:ensure-generic-function 'execution_state-val :lambda-list '(m))
(cl:defmethod execution_state-val ((m <RAPIDTaskState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:execution_state-val is deprecated.  Use abb_robot_msgs-msg:execution_state instead.")
  (execution_state m))

(cl:ensure-generic-function 'motion_task-val :lambda-list '(m))
(cl:defmethod motion_task-val ((m <RAPIDTaskState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:motion_task-val is deprecated.  Use abb_robot_msgs-msg:motion_task instead.")
  (motion_task m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<RAPIDTaskState>)))
    "Constants for message type '<RAPIDTaskState>"
  '((:EXECUTION_STATE_UNKNOWN . 1)
    (:EXECUTION_STATE_READY . 2)
    (:EXECUTION_STATE_STOPPED . 3)
    (:EXECUTION_STATE_STARTED . 4)
    (:EXECUTION_STATE_UNINITIALIZED . 5))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'RAPIDTaskState)))
    "Constants for message type 'RAPIDTaskState"
  '((:EXECUTION_STATE_UNKNOWN . 1)
    (:EXECUTION_STATE_READY . 2)
    (:EXECUTION_STATE_STOPPED . 3)
    (:EXECUTION_STATE_STARTED . 4)
    (:EXECUTION_STATE_UNINITIALIZED . 5))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RAPIDTaskState>) ostream)
  "Serializes a message object of type '<RAPIDTaskState>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'name))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'activated) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'execution_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'motion_task) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RAPIDTaskState>) istream)
  "Deserializes a message object of type '<RAPIDTaskState>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'activated) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'execution_state)) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'motion_task) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RAPIDTaskState>)))
  "Returns string type for a message object of type '<RAPIDTaskState>"
  "abb_robot_msgs/RAPIDTaskState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RAPIDTaskState)))
  "Returns string type for a message object of type 'RAPIDTaskState"
  "abb_robot_msgs/RAPIDTaskState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RAPIDTaskState>)))
  "Returns md5sum for a message object of type '<RAPIDTaskState>"
  "37a1a82a542dcaea43aef8b824e26614")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RAPIDTaskState)))
  "Returns md5sum for a message object of type 'RAPIDTaskState"
  "37a1a82a542dcaea43aef8b824e26614")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RAPIDTaskState>)))
  "Returns full string definition for message of type '<RAPIDTaskState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a RAPID task defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# RAPID execution states:~%uint8 EXECUTION_STATE_UNKNOWN       = 1~%uint8 EXECUTION_STATE_READY         = 2~%uint8 EXECUTION_STATE_STOPPED       = 3~%uint8 EXECUTION_STATE_STARTED       = 4~%uint8 EXECUTION_STATE_UNINITIALIZED = 5~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task's name.~%string name~%~%# The RAPID task's activated/deactivated status.~%bool activated~%~%# The RAPID task's execution state.~%uint8 execution_state~%~%# The RAPID task is a motion task.~%bool motion_task~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RAPIDTaskState)))
  "Returns full string definition for message of type 'RAPIDTaskState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a RAPID task defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# RAPID execution states:~%uint8 EXECUTION_STATE_UNKNOWN       = 1~%uint8 EXECUTION_STATE_READY         = 2~%uint8 EXECUTION_STATE_STOPPED       = 3~%uint8 EXECUTION_STATE_STARTED       = 4~%uint8 EXECUTION_STATE_UNINITIALIZED = 5~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task's name.~%string name~%~%# The RAPID task's activated/deactivated status.~%bool activated~%~%# The RAPID task's execution state.~%uint8 execution_state~%~%# The RAPID task is a motion task.~%bool motion_task~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RAPIDTaskState>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'name))
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RAPIDTaskState>))
  "Converts a ROS message object to a list"
  (cl:list 'RAPIDTaskState
    (cl:cons ':name (name msg))
    (cl:cons ':activated (activated msg))
    (cl:cons ':execution_state (execution_state msg))
    (cl:cons ':motion_task (motion_task msg))
))
