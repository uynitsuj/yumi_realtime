; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-msg)


;//! \htmlinclude RAPIDSymbolPath.msg.html

(cl:defclass <RAPIDSymbolPath> (roslisp-msg-protocol:ros-message)
  ((task
    :reader task
    :initarg :task
    :type cl:string
    :initform "")
   (module
    :reader module
    :initarg :module
    :type cl:string
    :initform "")
   (symbol
    :reader symbol
    :initarg :symbol
    :type cl:string
    :initform ""))
)

(cl:defclass RAPIDSymbolPath (<RAPIDSymbolPath>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RAPIDSymbolPath>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RAPIDSymbolPath)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-msg:<RAPIDSymbolPath> is deprecated: use abb_robot_msgs-msg:RAPIDSymbolPath instead.")))

(cl:ensure-generic-function 'task-val :lambda-list '(m))
(cl:defmethod task-val ((m <RAPIDSymbolPath>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:task-val is deprecated.  Use abb_robot_msgs-msg:task instead.")
  (task m))

(cl:ensure-generic-function 'module-val :lambda-list '(m))
(cl:defmethod module-val ((m <RAPIDSymbolPath>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:module-val is deprecated.  Use abb_robot_msgs-msg:module instead.")
  (module m))

(cl:ensure-generic-function 'symbol-val :lambda-list '(m))
(cl:defmethod symbol-val ((m <RAPIDSymbolPath>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:symbol-val is deprecated.  Use abb_robot_msgs-msg:symbol instead.")
  (symbol m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RAPIDSymbolPath>) ostream)
  "Serializes a message object of type '<RAPIDSymbolPath>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'task))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'task))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'module))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'module))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'symbol))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'symbol))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RAPIDSymbolPath>) istream)
  "Deserializes a message object of type '<RAPIDSymbolPath>"
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
      (cl:setf (cl:slot-value msg 'module) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'module) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'symbol) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'symbol) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RAPIDSymbolPath>)))
  "Returns string type for a message object of type '<RAPIDSymbolPath>"
  "abb_robot_msgs/RAPIDSymbolPath")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RAPIDSymbolPath)))
  "Returns string type for a message object of type 'RAPIDSymbolPath"
  "abb_robot_msgs/RAPIDSymbolPath")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RAPIDSymbolPath>)))
  "Returns md5sum for a message object of type '<RAPIDSymbolPath>"
  "3214a5a49942fc04a524a160d9874980")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RAPIDSymbolPath)))
  "Returns md5sum for a message object of type 'RAPIDSymbolPath"
  "3214a5a49942fc04a524a160d9874980")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RAPIDSymbolPath>)))
  "Returns full string definition for message of type '<RAPIDSymbolPath>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RAPIDSymbolPath)))
  "Returns full string definition for message of type 'RAPIDSymbolPath"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the path to a RAPID~%#   symbol (e.g. a variable) defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).~%string task~%~%# Name of the RAPID module, where the symbol has been defined.~%string module~%~%# Name of the RAPID symbol in question.~%string symbol~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RAPIDSymbolPath>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'task))
     4 (cl:length (cl:slot-value msg 'module))
     4 (cl:length (cl:slot-value msg 'symbol))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RAPIDSymbolPath>))
  "Converts a ROS message object to a list"
  (cl:list 'RAPIDSymbolPath
    (cl:cons ':task (task msg))
    (cl:cons ':module (module msg))
    (cl:cons ':symbol (symbol msg))
))
