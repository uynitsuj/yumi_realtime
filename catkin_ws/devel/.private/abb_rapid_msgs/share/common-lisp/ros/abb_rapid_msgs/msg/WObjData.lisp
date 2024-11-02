; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude WObjData.msg.html

(cl:defclass <WObjData> (roslisp-msg-protocol:ros-message)
  ((robhold
    :reader robhold
    :initarg :robhold
    :type cl:boolean
    :initform cl:nil)
   (ufprog
    :reader ufprog
    :initarg :ufprog
    :type cl:boolean
    :initform cl:nil)
   (ufmec
    :reader ufmec
    :initarg :ufmec
    :type cl:string
    :initform "")
   (uframe
    :reader uframe
    :initarg :uframe
    :type abb_rapid_msgs-msg:Pose
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pose))
   (oframe
    :reader oframe
    :initarg :oframe
    :type abb_rapid_msgs-msg:Pose
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pose)))
)

(cl:defclass WObjData (<WObjData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <WObjData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'WObjData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<WObjData> is deprecated: use abb_rapid_msgs-msg:WObjData instead.")))

(cl:ensure-generic-function 'robhold-val :lambda-list '(m))
(cl:defmethod robhold-val ((m <WObjData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:robhold-val is deprecated.  Use abb_rapid_msgs-msg:robhold instead.")
  (robhold m))

(cl:ensure-generic-function 'ufprog-val :lambda-list '(m))
(cl:defmethod ufprog-val ((m <WObjData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:ufprog-val is deprecated.  Use abb_rapid_msgs-msg:ufprog instead.")
  (ufprog m))

(cl:ensure-generic-function 'ufmec-val :lambda-list '(m))
(cl:defmethod ufmec-val ((m <WObjData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:ufmec-val is deprecated.  Use abb_rapid_msgs-msg:ufmec instead.")
  (ufmec m))

(cl:ensure-generic-function 'uframe-val :lambda-list '(m))
(cl:defmethod uframe-val ((m <WObjData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:uframe-val is deprecated.  Use abb_rapid_msgs-msg:uframe instead.")
  (uframe m))

(cl:ensure-generic-function 'oframe-val :lambda-list '(m))
(cl:defmethod oframe-val ((m <WObjData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:oframe-val is deprecated.  Use abb_rapid_msgs-msg:oframe instead.")
  (oframe m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <WObjData>) ostream)
  "Serializes a message object of type '<WObjData>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'robhold) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'ufprog) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'ufmec))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'ufmec))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'uframe) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'oframe) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <WObjData>) istream)
  "Deserializes a message object of type '<WObjData>"
    (cl:setf (cl:slot-value msg 'robhold) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'ufprog) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'ufmec) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'ufmec) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'uframe) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'oframe) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<WObjData>)))
  "Returns string type for a message object of type '<WObjData>"
  "abb_rapid_msgs/WObjData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'WObjData)))
  "Returns string type for a message object of type 'WObjData"
  "abb_rapid_msgs/WObjData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<WObjData>)))
  "Returns md5sum for a message object of type '<WObjData>"
  "eb6fd21e8b900ca32b7fe6c07697ac07")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'WObjData)))
  "Returns md5sum for a message object of type 'WObjData"
  "eb6fd21e8b900ca32b7fe6c07697ac07")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<WObjData>)))
  "Returns full string definition for message of type '<WObjData>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"wobjdata\" (work object data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot in the actual~%# program task is holding the work object:~%# - TRUE: The robot is holding the work object,~%#         i.e. using a stationary tool.~%# - FALSE: The robot is not holding the work object,~%#          i.e. the robot is holding the tool.~%bool robhold~%~%# Defines whether or not a fixed user coordinate system is used:~%# - TRUE: Fixed user coordinate system.~%# - FALSE: Movable user coordinate system,~%#          i.e. coordinated external axes are used.~%#          Also to be used in a MultiMove system in~%#          semicoordinated or synchronized coordinated mode.~%bool ufprog~%~%# The mechanical unit with which the robot movements are coordinated.~%#~%# Only specified in the case of movable user~%# coordinate systems (ufprog is FALSE).~%string ufmec~%~%# The user coordinate system, i.e. the position~%# of the current work surface or fixture.~%Pose uframe~%~%# The object coordinate system, i.e. the position~%# of the current work object.~%Pose oframe~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'WObjData)))
  "Returns full string definition for message of type 'WObjData"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"wobjdata\" (work object data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Defines whether or not the robot in the actual~%# program task is holding the work object:~%# - TRUE: The robot is holding the work object,~%#         i.e. using a stationary tool.~%# - FALSE: The robot is not holding the work object,~%#          i.e. the robot is holding the tool.~%bool robhold~%~%# Defines whether or not a fixed user coordinate system is used:~%# - TRUE: Fixed user coordinate system.~%# - FALSE: Movable user coordinate system,~%#          i.e. coordinated external axes are used.~%#          Also to be used in a MultiMove system in~%#          semicoordinated or synchronized coordinated mode.~%bool ufprog~%~%# The mechanical unit with which the robot movements are coordinated.~%#~%# Only specified in the case of movable user~%# coordinate systems (ufprog is FALSE).~%string ufmec~%~%# The user coordinate system, i.e. the position~%# of the current work surface or fixture.~%Pose uframe~%~%# The object coordinate system, i.e. the position~%# of the current work object.~%Pose oframe~%~%================================================================================~%MSG: abb_rapid_msgs/Pose~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pose\" (coordinate transformations).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The displacement in position (x, y, and z) of the coordinate system.~%Pos trans~%~%# The rotation of the coordinate system.~%Orient rot~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <WObjData>))
  (cl:+ 0
     1
     1
     4 (cl:length (cl:slot-value msg 'ufmec))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'uframe))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'oframe))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <WObjData>))
  "Converts a ROS message object to a list"
  (cl:list 'WObjData
    (cl:cons ':robhold (robhold msg))
    (cl:cons ':ufprog (ufprog msg))
    (cl:cons ':ufmec (ufmec msg))
    (cl:cons ':uframe (uframe msg))
    (cl:cons ':oframe (oframe msg))
))
