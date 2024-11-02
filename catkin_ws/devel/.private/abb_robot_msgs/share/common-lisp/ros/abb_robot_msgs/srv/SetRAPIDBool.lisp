; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude SetRAPIDBool-request.msg.html

(cl:defclass <SetRAPIDBool-request> (roslisp-msg-protocol:ros-message)
  ((path
    :reader path
    :initarg :path
    :type abb_robot_msgs-msg:RAPIDSymbolPath
    :initform (cl:make-instance 'abb_robot_msgs-msg:RAPIDSymbolPath))
   (value
    :reader value
    :initarg :value
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetRAPIDBool-request (<SetRAPIDBool-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetRAPIDBool-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetRAPIDBool-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetRAPIDBool-request> is deprecated: use abb_robot_msgs-srv:SetRAPIDBool-request instead.")))

(cl:ensure-generic-function 'path-val :lambda-list '(m))
(cl:defmethod path-val ((m <SetRAPIDBool-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:path-val is deprecated.  Use abb_robot_msgs-srv:path instead.")
  (path m))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <SetRAPIDBool-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:value-val is deprecated.  Use abb_robot_msgs-srv:value instead.")
  (value m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetRAPIDBool-request>) ostream)
  "Serializes a message object of type '<SetRAPIDBool-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'path) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'value) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetRAPIDBool-request>) istream)
  "Deserializes a message object of type '<SetRAPIDBool-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'path) istream)
    (cl:setf (cl:slot-value msg 'value) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetRAPIDBool-request>)))
  "Returns string type for a service object of type '<SetRAPIDBool-request>"
  "abb_robot_msgs/SetRAPIDBoolRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDBool-request)))
  "Returns string type for a service object of type 'SetRAPIDBool-request"
  "abb_robot_msgs/SetRAPIDBoolRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetRAPIDBool-request>)))
  "Returns md5sum for a message object of type '<SetRAPIDBool-request>"
  "932d876676bb80924a225c457e4d06a3")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetRAPIDBool-request)))
  "Returns md5sum for a message object of type 'SetRAPIDBool-request"
  "932d876676bb80924a225c457e4d06a3")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetRAPIDBool-request>)))
  "Returns full string definition for message of type '<SetRAPIDBool-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the value~%#   of a RAPID 'bool' symbol (e.g. a variable) defined in an ABB robot~%#   controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%# New value of the RAPID symbol.~%bool value~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetRAPIDBool-request)))
  "Returns full string definition for message of type 'SetRAPIDBool-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the value~%#   of a RAPID 'bool' symbol (e.g. a variable) defined in an ABB robot~%#   controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Path to the targeted RAPID symbol.~%RAPIDSymbolPath path~%~%# New value of the RAPID symbol.~%bool value~%~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDSymbolPath~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetRAPIDBool-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'path))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetRAPIDBool-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetRAPIDBool-request
    (cl:cons ':path (path msg))
    (cl:cons ':value (value msg))
))
;//! \htmlinclude SetRAPIDBool-response.msg.html

(cl:defclass <SetRAPIDBool-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetRAPIDBool-response (<SetRAPIDBool-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetRAPIDBool-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetRAPIDBool-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetRAPIDBool-response> is deprecated: use abb_robot_msgs-srv:SetRAPIDBool-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetRAPIDBool-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetRAPIDBool-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetRAPIDBool-response>) ostream)
  "Serializes a message object of type '<SetRAPIDBool-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetRAPIDBool-response>) istream)
  "Deserializes a message object of type '<SetRAPIDBool-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetRAPIDBool-response>)))
  "Returns string type for a service object of type '<SetRAPIDBool-response>"
  "abb_robot_msgs/SetRAPIDBoolResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDBool-response)))
  "Returns string type for a service object of type 'SetRAPIDBool-response"
  "abb_robot_msgs/SetRAPIDBoolResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetRAPIDBool-response>)))
  "Returns md5sum for a message object of type '<SetRAPIDBool-response>"
  "932d876676bb80924a225c457e4d06a3")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetRAPIDBool-response)))
  "Returns md5sum for a message object of type 'SetRAPIDBool-response"
  "932d876676bb80924a225c457e4d06a3")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetRAPIDBool-response>)))
  "Returns full string definition for message of type '<SetRAPIDBool-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetRAPIDBool-response)))
  "Returns full string definition for message of type 'SetRAPIDBool-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetRAPIDBool-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetRAPIDBool-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetRAPIDBool-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetRAPIDBool)))
  'SetRAPIDBool-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetRAPIDBool)))
  'SetRAPIDBool-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetRAPIDBool)))
  "Returns string type for a service object of type '<SetRAPIDBool>"
  "abb_robot_msgs/SetRAPIDBool")