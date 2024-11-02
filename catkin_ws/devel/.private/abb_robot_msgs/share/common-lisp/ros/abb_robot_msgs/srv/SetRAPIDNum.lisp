; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude SetRAPIDNum-request.msg.html

(cl:defclass <SetRAPIDNum-request> (roslisp-msg-protocol:ros-message)
  ((path
    :reader path
    :initarg :path
    :type abb_robot_msgs-msg:RAPIDSymbolPath
    :initform (cl:make-instance 'abb_robot_msgs-msg:RAPIDSymbolPath))
   (value
    :reader value
    :initarg :value
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetRAPIDNum-request (<SetRAPIDNum-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetRAPIDNum-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetRAPIDNum-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetRAPIDNum-request> is deprecated: use abb_robot_msgs-srv:SetRAPIDNum-request instead.")))

(cl:ensure-generic-function 'path-val :lambda-list '(m))
(cl:defmethod path-val ((m <SetRAPIDNum-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:path-val is deprecated.  Use abb_robot_msgs-srv:path instead.")
  (path m))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <SetRAPIDNum-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:value-val is deprecated.  Use abb_robot_msgs-srv:value instead.")
  (value m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetRAPIDNum-request>) ostream)
  "Serializes a message object of type '<SetRAPIDNum-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'path) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetRAPIDNum-request>) istream)
  "Deserializes a message object of type '<SetRAPIDNum-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'path) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'value) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetRAPIDNum-request>)))
  "Returns string type for a service object of type '<SetRAPIDNum-request>"
  "abb_robot_msgs/SetRAPIDNumRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDNum-request)))
  "Returns string type for a service object of type 'SetRAPIDNum-request"
  "abb_robot_msgs/SetRAPIDNumRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetRAPIDNum-request>)))
  "Returns md5sum for a message object of type '<SetRAPIDNum-request>"
  "4108cd886758eb287b6fb706804d2b63")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetRAPIDNum-request)))
  "Returns md5sum for a message object of type 'SetRAPIDNum-request"
  "4108cd886758eb287b6fb706804d2b63")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetRAPIDNum-request>)))
  "Returns full string definition for message of type '<SetRAPIDNum-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the value~%#   of a RAPID 'num' symbol (e.g. a variable) defined in an ABB robot controller~%#   system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%# New value of the RAPID symbol.~%float32 value~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetRAPIDNum-request)))
  "Returns full string definition for message of type 'SetRAPIDNum-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the value~%#   of a RAPID 'num' symbol (e.g. a variable) defined in an ABB robot controller~%#   system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%# New value of the RAPID symbol.~%float32 value~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetRAPIDNum-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'path))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetRAPIDNum-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetRAPIDNum-request
    (cl:cons ':path (path msg))
    (cl:cons ':value (value msg))
))
;//! \htmlinclude SetRAPIDNum-response.msg.html

(cl:defclass <SetRAPIDNum-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetRAPIDNum-response (<SetRAPIDNum-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetRAPIDNum-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetRAPIDNum-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetRAPIDNum-response> is deprecated: use abb_robot_msgs-srv:SetRAPIDNum-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetRAPIDNum-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetRAPIDNum-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetRAPIDNum-response>) ostream)
  "Serializes a message object of type '<SetRAPIDNum-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetRAPIDNum-response>) istream)
  "Deserializes a message object of type '<SetRAPIDNum-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetRAPIDNum-response>)))
  "Returns string type for a service object of type '<SetRAPIDNum-response>"
  "abb_robot_msgs/SetRAPIDNumResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDNum-response)))
  "Returns string type for a service object of type 'SetRAPIDNum-response"
  "abb_robot_msgs/SetRAPIDNumResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetRAPIDNum-response>)))
  "Returns md5sum for a message object of type '<SetRAPIDNum-response>"
  "4108cd886758eb287b6fb706804d2b63")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetRAPIDNum-response)))
  "Returns md5sum for a message object of type 'SetRAPIDNum-response"
  "4108cd886758eb287b6fb706804d2b63")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetRAPIDNum-response>)))
  "Returns full string definition for message of type '<SetRAPIDNum-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetRAPIDNum-response)))
  "Returns full string definition for message of type 'SetRAPIDNum-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetRAPIDNum-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetRAPIDNum-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetRAPIDNum-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetRAPIDNum)))
  'SetRAPIDNum-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetRAPIDNum)))
  'SetRAPIDNum-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDNum)))
  "Returns string type for a service object of type '<SetRAPIDNum>"
  "abb_robot_msgs/SetRAPIDNum")