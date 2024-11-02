; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude GetRAPIDNum-request.msg.html

(cl:defclass <GetRAPIDNum-request> (roslisp-msg-protocol:ros-message)
  ((path
    :reader path
    :initarg :path
    :type abb_robot_msgs-msg:RAPIDSymbolPath
    :initform (cl:make-instance 'abb_robot_msgs-msg:RAPIDSymbolPath)))
)

(cl:defclass GetRAPIDNum-request (<GetRAPIDNum-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetRAPIDNum-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetRAPIDNum-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetRAPIDNum-request> is deprecated: use abb_robot_msgs-srv:GetRAPIDNum-request instead.")))

(cl:ensure-generic-function 'path-val :lambda-list '(m))
(cl:defmethod path-val ((m <GetRAPIDNum-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:path-val is deprecated.  Use abb_robot_msgs-srv:path instead.")
  (path m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetRAPIDNum-request>) ostream)
  "Serializes a message object of type '<GetRAPIDNum-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'path) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetRAPIDNum-request>) istream)
  "Deserializes a message object of type '<GetRAPIDNum-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'path) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetRAPIDNum-request>)))
  "Returns string type for a service object of type '<GetRAPIDNum-request>"
  "abb_robot_msgs/GetRAPIDNumRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRAPIDNum-request)))
  "Returns string type for a service object of type 'GetRAPIDNum-request"
  "abb_robot_msgs/GetRAPIDNumRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetRAPIDNum-request>)))
  "Returns md5sum for a message object of type '<GetRAPIDNum-request>"
  "50b6604c67513df1c689dbe2ae540bb1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetRAPIDNum-request)))
  "Returns md5sum for a message object of type 'GetRAPIDNum-request"
  "50b6604c67513df1c689dbe2ae540bb1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetRAPIDNum-request>)))
  "Returns full string definition for message of type '<GetRAPIDNum-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the value~%#   of a RAPID 'num' symbol (e.g. a variable) defined in an ABB robot controller~%#   system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetRAPIDNum-request)))
  "Returns full string definition for message of type 'GetRAPIDNum-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the value~%#   of a RAPID 'num' symbol (e.g. a variable) defined in an ABB robot controller~%#   system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetRAPIDNum-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'path))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetRAPIDNum-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetRAPIDNum-request
    (cl:cons ':path (path msg))
))
;//! \htmlinclude GetRAPIDNum-response.msg.html

(cl:defclass <GetRAPIDNum-response> (roslisp-msg-protocol:ros-message)
  ((value
    :reader value
    :initarg :value
    :type cl:float
    :initform 0.0)
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

(cl:defclass GetRAPIDNum-response (<GetRAPIDNum-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetRAPIDNum-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetRAPIDNum-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetRAPIDNum-response> is deprecated: use abb_robot_msgs-srv:GetRAPIDNum-response instead.")))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <GetRAPIDNum-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:value-val is deprecated.  Use abb_robot_msgs-srv:value instead.")
  (value m))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <GetRAPIDNum-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <GetRAPIDNum-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetRAPIDNum-response>) ostream)
  "Serializes a message object of type '<GetRAPIDNum-response>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetRAPIDNum-response>) istream)
  "Deserializes a message object of type '<GetRAPIDNum-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'value) (roslisp-utils:decode-single-float-bits bits)))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetRAPIDNum-response>)))
  "Returns string type for a service object of type '<GetRAPIDNum-response>"
  "abb_robot_msgs/GetRAPIDNumResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRAPIDNum-response)))
  "Returns string type for a service object of type 'GetRAPIDNum-response"
  "abb_robot_msgs/GetRAPIDNumResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetRAPIDNum-response>)))
  "Returns md5sum for a message object of type '<GetRAPIDNum-response>"
  "50b6604c67513df1c689dbe2ae540bb1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetRAPIDNum-response)))
  "Returns md5sum for a message object of type 'GetRAPIDNum-response"
  "50b6604c67513df1c689dbe2ae540bb1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetRAPIDNum-response>)))
  "Returns full string definition for message of type '<GetRAPIDNum-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Value of the RAPID symbol.~%float32 value~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetRAPIDNum-response)))
  "Returns full string definition for message of type 'GetRAPIDNum-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Value of the RAPID symbol.~%float32 value~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetRAPIDNum-response>))
  (cl:+ 0
     4
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetRAPIDNum-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetRAPIDNum-response
    (cl:cons ':value (value msg))
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetRAPIDNum)))
  'GetRAPIDNum-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetRAPIDNum)))
  'GetRAPIDNum-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRAPIDNum)))
  "Returns string type for a service object of type '<GetRAPIDNum>"
  "abb_robot_msgs/GetRAPIDNum")