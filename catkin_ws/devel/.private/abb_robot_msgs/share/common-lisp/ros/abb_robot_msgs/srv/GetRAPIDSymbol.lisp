; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude GetRAPIDSymbol-request.msg.html

(cl:defclass <GetRAPIDSymbol-request> (roslisp-msg-protocol:ros-message)
  ((path
    :reader path
    :initarg :path
    :type abb_robot_msgs-msg:RAPIDSymbolPath
    :initform (cl:make-instance 'abb_robot_msgs-msg:RAPIDSymbolPath)))
)

(cl:defclass GetRAPIDSymbol-request (<GetRAPIDSymbol-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetRAPIDSymbol-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetRAPIDSymbol-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetRAPIDSymbol-request> is deprecated: use abb_robot_msgs-srv:GetRAPIDSymbol-request instead.")))

(cl:ensure-generic-function 'path-val :lambda-list '(m))
(cl:defmethod path-val ((m <GetRAPIDSymbol-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:path-val is deprecated.  Use abb_robot_msgs-srv:path instead.")
  (path m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetRAPIDSymbol-request>) ostream)
  "Serializes a message object of type '<GetRAPIDSymbol-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'path) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetRAPIDSymbol-request>) istream)
  "Deserializes a message object of type '<GetRAPIDSymbol-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'path) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetRAPIDSymbol-request>)))
  "Returns string type for a service object of type '<GetRAPIDSymbol-request>"
  "abb_robot_msgs/GetRAPIDSymbolRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRAPIDSymbol-request)))
  "Returns string type for a service object of type 'GetRAPIDSymbol-request"
  "abb_robot_msgs/GetRAPIDSymbolRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetRAPIDSymbol-request>)))
  "Returns md5sum for a message object of type '<GetRAPIDSymbol-request>"
  "bde39a2871b891ab4e4dba6d1b2afbe4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetRAPIDSymbol-request)))
  "Returns md5sum for a message object of type 'GetRAPIDSymbol-request"
  "bde39a2871b891ab4e4dba6d1b2afbe4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetRAPIDSymbol-request>)))
  "Returns full string definition for message of type '<GetRAPIDSymbol-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the value~%#   of a RAPID symbol (e.g. a variable) defined in an ABB robot controller~%#   system.~%#~%# Note:~%#   The RAPID symbol's value will be returned in raw text format.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetRAPIDSymbol-request)))
  "Returns full string definition for message of type 'GetRAPIDSymbol-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the value~%#   of a RAPID symbol (e.g. a variable) defined in an ABB robot controller~%#   system.~%#~%# Note:~%#   The RAPID symbol's value will be returned in raw text format.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetRAPIDSymbol-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'path))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetRAPIDSymbol-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetRAPIDSymbol-request
    (cl:cons ':path (path msg))
))
;//! \htmlinclude GetRAPIDSymbol-response.msg.html

(cl:defclass <GetRAPIDSymbol-response> (roslisp-msg-protocol:ros-message)
  ((value
    :reader value
    :initarg :value
    :type cl:string
    :initform "")
   (result_code
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

(cl:defclass GetRAPIDSymbol-response (<GetRAPIDSymbol-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetRAPIDSymbol-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetRAPIDSymbol-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetRAPIDSymbol-response> is deprecated: use abb_robot_msgs-srv:GetRAPIDSymbol-response instead.")))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <GetRAPIDSymbol-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:value-val is deprecated.  Use abb_robot_msgs-srv:value instead.")
  (value m))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <GetRAPIDSymbol-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <GetRAPIDSymbol-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetRAPIDSymbol-response>) ostream)
  "Serializes a message object of type '<GetRAPIDSymbol-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'value))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetRAPIDSymbol-response>) istream)
  "Deserializes a message object of type '<GetRAPIDSymbol-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'value) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'value) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetRAPIDSymbol-response>)))
  "Returns string type for a service object of type '<GetRAPIDSymbol-response>"
  "abb_robot_msgs/GetRAPIDSymbolResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRAPIDSymbol-response)))
  "Returns string type for a service object of type 'GetRAPIDSymbol-response"
  "abb_robot_msgs/GetRAPIDSymbolResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetRAPIDSymbol-response>)))
  "Returns md5sum for a message object of type '<GetRAPIDSymbol-response>"
  "bde39a2871b891ab4e4dba6d1b2afbe4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetRAPIDSymbol-response)))
  "Returns md5sum for a message object of type 'GetRAPIDSymbol-response"
  "bde39a2871b891ab4e4dba6d1b2afbe4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetRAPIDSymbol-response>)))
  "Returns full string definition for message of type '<GetRAPIDSymbol-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Value (in raw text format) of the RAPID symbol.~%string value~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetRAPIDSymbol-response)))
  "Returns full string definition for message of type 'GetRAPIDSymbol-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Value (in raw text format) of the RAPID symbol.~%string value~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetRAPIDSymbol-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'value))
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetRAPIDSymbol-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetRAPIDSymbol-response
    (cl:cons ':value (value msg))
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetRAPIDSymbol)))
  'GetRAPIDSymbol-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetRAPIDSymbol)))
  'GetRAPIDSymbol-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRAPIDSymbol)))
  "Returns string type for a service object of type '<GetRAPIDSymbol>"
  "abb_robot_msgs/GetRAPIDSymbol")