; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-srv)


;//! \htmlinclude SetRAPIDRoutine-request.msg.html

(cl:defclass <SetRAPIDRoutine-request> (roslisp-msg-protocol:ros-message)
  ((task
    :reader task
    :initarg :task
    :type cl:string
    :initform "")
   (routine
    :reader routine
    :initarg :routine
    :type cl:string
    :initform ""))
)

(cl:defclass SetRAPIDRoutine-request (<SetRAPIDRoutine-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetRAPIDRoutine-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetRAPIDRoutine-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-srv:<SetRAPIDRoutine-request> is deprecated: use abb_rapid_sm_addin_msgs-srv:SetRAPIDRoutine-request instead.")))

(cl:ensure-generic-function 'task-val :lambda-list '(m))
(cl:defmethod task-val ((m <SetRAPIDRoutine-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:task-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:task instead.")
  (task m))

(cl:ensure-generic-function 'routine-val :lambda-list '(m))
(cl:defmethod routine-val ((m <SetRAPIDRoutine-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:routine-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:routine instead.")
  (routine m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetRAPIDRoutine-request>) ostream)
  "Serializes a message object of type '<SetRAPIDRoutine-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'task))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'task))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'routine))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'routine))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetRAPIDRoutine-request>) istream)
  "Deserializes a message object of type '<SetRAPIDRoutine-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'task) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'task) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'routine) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'routine) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetRAPIDRoutine-request>)))
  "Returns string type for a service object of type '<SetRAPIDRoutine-request>"
  "abb_rapid_sm_addin_msgs/SetRAPIDRoutineRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDRoutine-request)))
  "Returns string type for a service object of type 'SetRAPIDRoutine-request"
  "abb_rapid_sm_addin_msgs/SetRAPIDRoutineRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetRAPIDRoutine-request>)))
  "Returns md5sum for a message object of type '<SetRAPIDRoutine-request>"
  "652cc4087927fd3e9d05a7309ce84794")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetRAPIDRoutine-request)))
  "Returns md5sum for a message object of type 'SetRAPIDRoutine-request"
  "652cc4087927fd3e9d05a7309ce84794")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetRAPIDRoutine-request>)))
  "Returns full string definition for message of type '<SetRAPIDRoutine-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the value~%#   of a predefined RAPID variable for specifying a custom RAPID routine to run.~%#~%# Note:~%#   This variable is defined in each RAPID motion task, in ABB robot~%#   controller systems, that are using the RobotWare \"StateMachine Add-In\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string task~%~%# Desired custom RAPID routine to run.~%string routine~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetRAPIDRoutine-request)))
  "Returns full string definition for message of type 'SetRAPIDRoutine-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the value~%#   of a predefined RAPID variable for specifying a custom RAPID routine to run.~%#~%# Note:~%#   This variable is defined in each RAPID motion task, in ABB robot~%#   controller systems, that are using the RobotWare \"StateMachine Add-In\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string task~%~%# Desired custom RAPID routine to run.~%string routine~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetRAPIDRoutine-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'task))
     4 (cl:length (cl:slot-value msg 'routine))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetRAPIDRoutine-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetRAPIDRoutine-request
    (cl:cons ':task (task msg))
    (cl:cons ':routine (routine msg))
))
;//! \htmlinclude SetRAPIDRoutine-response.msg.html

(cl:defclass <SetRAPIDRoutine-response> (roslisp-msg-protocol:ros-message)
  ((result_code
    :reader result_code
    :initarg :result_code
    :type cl:fixnum
    :initform 0)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass SetRAPIDRoutine-response (<SetRAPIDRoutine-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetRAPIDRoutine-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetRAPIDRoutine-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-srv:<SetRAPIDRoutine-response> is deprecated: use abb_rapid_sm_addin_msgs-srv:SetRAPIDRoutine-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetRAPIDRoutine-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:result_code-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetRAPIDRoutine-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:message-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetRAPIDRoutine-response>) ostream)
  "Serializes a message object of type '<SetRAPIDRoutine-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetRAPIDRoutine-response>) istream)
  "Deserializes a message object of type '<SetRAPIDRoutine-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) (cl:read-byte istream))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetRAPIDRoutine-response>)))
  "Returns string type for a service object of type '<SetRAPIDRoutine-response>"
  "abb_rapid_sm_addin_msgs/SetRAPIDRoutineResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDRoutine-response)))
  "Returns string type for a service object of type 'SetRAPIDRoutine-response"
  "abb_rapid_sm_addin_msgs/SetRAPIDRoutineResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetRAPIDRoutine-response>)))
  "Returns md5sum for a message object of type '<SetRAPIDRoutine-response>"
  "652cc4087927fd3e9d05a7309ce84794")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetRAPIDRoutine-response)))
  "Returns md5sum for a message object of type 'SetRAPIDRoutine-response"
  "652cc4087927fd3e9d05a7309ce84794")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetRAPIDRoutine-response>)))
  "Returns full string definition for message of type '<SetRAPIDRoutine-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetRAPIDRoutine-response)))
  "Returns full string definition for message of type 'SetRAPIDRoutine-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetRAPIDRoutine-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetRAPIDRoutine-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetRAPIDRoutine-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetRAPIDRoutine)))
  'SetRAPIDRoutine-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetRAPIDRoutine)))
  'SetRAPIDRoutine-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDRoutine)))
  "Returns string type for a service object of type '<SetRAPIDRoutine>"
  "abb_rapid_sm_addin_msgs/SetRAPIDRoutine")