; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude GetFileContents-request.msg.html

(cl:defclass <GetFileContents-request> (roslisp-msg-protocol:ros-message)
  ((filename
    :reader filename
    :initarg :filename
    :type cl:string
    :initform ""))
)

(cl:defclass GetFileContents-request (<GetFileContents-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetFileContents-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetFileContents-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetFileContents-request> is deprecated: use abb_robot_msgs-srv:GetFileContents-request instead.")))

(cl:ensure-generic-function 'filename-val :lambda-list '(m))
(cl:defmethod filename-val ((m <GetFileContents-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:filename-val is deprecated.  Use abb_robot_msgs-srv:filename instead.")
  (filename m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetFileContents-request>) ostream)
  "Serializes a message object of type '<GetFileContents-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'filename))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'filename))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetFileContents-request>) istream)
  "Deserializes a message object of type '<GetFileContents-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'filename) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'filename) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetFileContents-request>)))
  "Returns string type for a service object of type '<GetFileContents-request>"
  "abb_robot_msgs/GetFileContentsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetFileContents-request)))
  "Returns string type for a service object of type 'GetFileContents-request"
  "abb_robot_msgs/GetFileContentsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetFileContents-request>)))
  "Returns md5sum for a message object of type '<GetFileContents-request>"
  "4715c1a4564cfe2f12eb2de6ae62ccaa")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetFileContents-request)))
  "Returns md5sum for a message object of type 'GetFileContents-request"
  "4715c1a4564cfe2f12eb2de6ae62ccaa")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetFileContents-request>)))
  "Returns full string definition for message of type '<GetFileContents-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the~%#   contents of a file existing in an ABB robot controller system.~%#~%# Note:~%#   The file is assumed to be located in the system's home directory.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Filename (including file extension) of the targeted file.~%string filename~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetFileContents-request)))
  "Returns full string definition for message of type 'GetFileContents-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the~%#   contents of a file existing in an ABB robot controller system.~%#~%# Note:~%#   The file is assumed to be located in the system's home directory.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Filename (including file extension) of the targeted file.~%string filename~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetFileContents-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'filename))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetFileContents-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetFileContents-request
    (cl:cons ':filename (filename msg))
))
;//! \htmlinclude GetFileContents-response.msg.html

(cl:defclass <GetFileContents-response> (roslisp-msg-protocol:ros-message)
  ((contents
    :reader contents
    :initarg :contents
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

(cl:defclass GetFileContents-response (<GetFileContents-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetFileContents-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetFileContents-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetFileContents-response> is deprecated: use abb_robot_msgs-srv:GetFileContents-response instead.")))

(cl:ensure-generic-function 'contents-val :lambda-list '(m))
(cl:defmethod contents-val ((m <GetFileContents-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:contents-val is deprecated.  Use abb_robot_msgs-srv:contents instead.")
  (contents m))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <GetFileContents-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <GetFileContents-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetFileContents-response>) ostream)
  "Serializes a message object of type '<GetFileContents-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'contents))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'contents))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetFileContents-response>) istream)
  "Deserializes a message object of type '<GetFileContents-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'contents) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'contents) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetFileContents-response>)))
  "Returns string type for a service object of type '<GetFileContents-response>"
  "abb_robot_msgs/GetFileContentsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetFileContents-response)))
  "Returns string type for a service object of type 'GetFileContents-response"
  "abb_robot_msgs/GetFileContentsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetFileContents-response>)))
  "Returns md5sum for a message object of type '<GetFileContents-response>"
  "4715c1a4564cfe2f12eb2de6ae62ccaa")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetFileContents-response)))
  "Returns md5sum for a message object of type 'GetFileContents-response"
  "4715c1a4564cfe2f12eb2de6ae62ccaa")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetFileContents-response>)))
  "Returns full string definition for message of type '<GetFileContents-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Contents of the file.~%string contents~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetFileContents-response)))
  "Returns full string definition for message of type 'GetFileContents-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Contents of the file.~%string contents~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetFileContents-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'contents))
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetFileContents-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetFileContents-response
    (cl:cons ':contents (contents msg))
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetFileContents)))
  'GetFileContents-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetFileContents)))
  'GetFileContents-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetFileContents)))
  "Returns string type for a service object of type '<GetFileContents>"
  "abb_robot_msgs/GetFileContents")