; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude SetFileContents-request.msg.html

(cl:defclass <SetFileContents-request> (roslisp-msg-protocol:ros-message)
  ((filename
    :reader filename
    :initarg :filename
    :type cl:string
    :initform "")
   (contents
    :reader contents
    :initarg :contents
    :type cl:string
    :initform ""))
)

(cl:defclass SetFileContents-request (<SetFileContents-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetFileContents-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetFileContents-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetFileContents-request> is deprecated: use abb_robot_msgs-srv:SetFileContents-request instead.")))

(cl:ensure-generic-function 'filename-val :lambda-list '(m))
(cl:defmethod filename-val ((m <SetFileContents-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:filename-val is deprecated.  Use abb_robot_msgs-srv:filename instead.")
  (filename m))

(cl:ensure-generic-function 'contents-val :lambda-list '(m))
(cl:defmethod contents-val ((m <SetFileContents-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:contents-val is deprecated.  Use abb_robot_msgs-srv:contents instead.")
  (contents m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetFileContents-request>) ostream)
  "Serializes a message object of type '<SetFileContents-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'filename))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'filename))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'contents))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'contents))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetFileContents-request>) istream)
  "Deserializes a message object of type '<SetFileContents-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'filename) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'filename) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'contents) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'contents) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetFileContents-request>)))
  "Returns string type for a service object of type '<SetFileContents-request>"
  "abb_robot_msgs/SetFileContentsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetFileContents-request)))
  "Returns string type for a service object of type 'SetFileContents-request"
  "abb_robot_msgs/SetFileContentsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetFileContents-request>)))
  "Returns md5sum for a message object of type '<SetFileContents-request>"
  "b9fd3989a30f4c529cc017e54ab4c80f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetFileContents-request)))
  "Returns md5sum for a message object of type 'SetFileContents-request"
  "b9fd3989a30f4c529cc017e54ab4c80f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetFileContents-request>)))
  "Returns full string definition for message of type '<SetFileContents-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the~%#   contents of a file in an ABB robot controller system.~%#~%# Note:~%#   The file will be placed in the system's home directory. If the file~%#   doesn't exist, then it will be created, otherwise it will be overwritten.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Filename (including file extension) for the file in the system.~%string filename~%~%# New contents for the file.~%string contents~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetFileContents-request)))
  "Returns full string definition for message of type 'SetFileContents-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the~%#   contents of a file in an ABB robot controller system.~%#~%# Note:~%#   The file will be placed in the system's home directory. If the file~%#   doesn't exist, then it will be created, otherwise it will be overwritten.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Filename (including file extension) for the file in the system.~%string filename~%~%# New contents for the file.~%string contents~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetFileContents-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'filename))
     4 (cl:length (cl:slot-value msg 'contents))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetFileContents-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetFileContents-request
    (cl:cons ':filename (filename msg))
    (cl:cons ':contents (contents msg))
))
;//! \htmlinclude SetFileContents-response.msg.html

(cl:defclass <SetFileContents-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetFileContents-response (<SetFileContents-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetFileContents-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetFileContents-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetFileContents-response> is deprecated: use abb_robot_msgs-srv:SetFileContents-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetFileContents-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetFileContents-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetFileContents-response>) ostream)
  "Serializes a message object of type '<SetFileContents-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetFileContents-response>) istream)
  "Deserializes a message object of type '<SetFileContents-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetFileContents-response>)))
  "Returns string type for a service object of type '<SetFileContents-response>"
  "abb_robot_msgs/SetFileContentsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetFileContents-response)))
  "Returns string type for a service object of type 'SetFileContents-response"
  "abb_robot_msgs/SetFileContentsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetFileContents-response>)))
  "Returns md5sum for a message object of type '<SetFileContents-response>"
  "b9fd3989a30f4c529cc017e54ab4c80f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetFileContents-response)))
  "Returns md5sum for a message object of type 'SetFileContents-response"
  "b9fd3989a30f4c529cc017e54ab4c80f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetFileContents-response>)))
  "Returns full string definition for message of type '<SetFileContents-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetFileContents-response)))
  "Returns full string definition for message of type 'SetFileContents-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetFileContents-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetFileContents-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetFileContents-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetFileContents)))
  'SetFileContents-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetFileContents)))
  'SetFileContents-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetFileContents)))
  "Returns string type for a service object of type '<SetFileContents>"
  "abb_robot_msgs/SetFileContents")