; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude LoadData.msg.html

(cl:defclass <LoadData> (roslisp-msg-protocol:ros-message)
  ((mass
    :reader mass
    :initarg :mass
    :type cl:float
    :initform 0.0)
   (cog
    :reader cog
    :initarg :cog
    :type abb_rapid_msgs-msg:Pos
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Pos))
   (aom
    :reader aom
    :initarg :aom
    :type abb_rapid_msgs-msg:Orient
    :initform (cl:make-instance 'abb_rapid_msgs-msg:Orient))
   (ix
    :reader ix
    :initarg :ix
    :type cl:float
    :initform 0.0)
   (iy
    :reader iy
    :initarg :iy
    :type cl:float
    :initform 0.0)
   (iz
    :reader iz
    :initarg :iz
    :type cl:float
    :initform 0.0))
)

(cl:defclass LoadData (<LoadData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LoadData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LoadData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<LoadData> is deprecated: use abb_rapid_msgs-msg:LoadData instead.")))

(cl:ensure-generic-function 'mass-val :lambda-list '(m))
(cl:defmethod mass-val ((m <LoadData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:mass-val is deprecated.  Use abb_rapid_msgs-msg:mass instead.")
  (mass m))

(cl:ensure-generic-function 'cog-val :lambda-list '(m))
(cl:defmethod cog-val ((m <LoadData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:cog-val is deprecated.  Use abb_rapid_msgs-msg:cog instead.")
  (cog m))

(cl:ensure-generic-function 'aom-val :lambda-list '(m))
(cl:defmethod aom-val ((m <LoadData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:aom-val is deprecated.  Use abb_rapid_msgs-msg:aom instead.")
  (aom m))

(cl:ensure-generic-function 'ix-val :lambda-list '(m))
(cl:defmethod ix-val ((m <LoadData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:ix-val is deprecated.  Use abb_rapid_msgs-msg:ix instead.")
  (ix m))

(cl:ensure-generic-function 'iy-val :lambda-list '(m))
(cl:defmethod iy-val ((m <LoadData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:iy-val is deprecated.  Use abb_rapid_msgs-msg:iy instead.")
  (iy m))

(cl:ensure-generic-function 'iz-val :lambda-list '(m))
(cl:defmethod iz-val ((m <LoadData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:iz-val is deprecated.  Use abb_rapid_msgs-msg:iz instead.")
  (iz m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LoadData>) ostream)
  "Serializes a message object of type '<LoadData>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'mass))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'cog) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'aom) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ix))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'iy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'iz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LoadData>) istream)
  "Deserializes a message object of type '<LoadData>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mass) (roslisp-utils:decode-single-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'cog) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'aom) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ix) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'iy) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'iz) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LoadData>)))
  "Returns string type for a message object of type '<LoadData>"
  "abb_rapid_msgs/LoadData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LoadData)))
  "Returns string type for a message object of type 'LoadData"
  "abb_rapid_msgs/LoadData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LoadData>)))
  "Returns md5sum for a message object of type '<LoadData>"
  "dfe575484752eff5e837f54d2fd939e4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LoadData)))
  "Returns md5sum for a message object of type 'LoadData"
  "dfe575484752eff5e837f54d2fd939e4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LoadData>)))
  "Returns full string definition for message of type '<LoadData>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"loaddata\" (load data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mass (weight) of the load in kg.~%float32 mass~%~%# The center of gravity of the payload, expressed in mm,~%# in the tool coordinate system if the robot is holding the tool.~%#~%# If a stationary tool is used then the center of gravity for the payload~%# held by the gripper is expressed in the object frame of the work object~%# coordinate system moved by the robot.~%Pos cog~%~%# The orientation of the axes of moment.~%#~%# These are the principal axes of the payload~%# moment of inertia with origin in cog.~%#~%# If the robot is holding the tool, the axes of moment~%# are expressed in the tool coordinate system.~%Orient aom~%~%# The moment of inertia of the load around the x-, y- and z-axis,~%# expressed in kgm^2.~%float32 ix~%float32 iy~%float32 iz~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LoadData)))
  "Returns full string definition for message of type 'LoadData"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"loaddata\" (load data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mass (weight) of the load in kg.~%float32 mass~%~%# The center of gravity of the payload, expressed in mm,~%# in the tool coordinate system if the robot is holding the tool.~%#~%# If a stationary tool is used then the center of gravity for the payload~%# held by the gripper is expressed in the object frame of the work object~%# coordinate system moved by the robot.~%Pos cog~%~%# The orientation of the axes of moment.~%#~%# These are the principal axes of the payload~%# moment of inertia with origin in cog.~%#~%# If the robot is holding the tool, the axes of moment~%# are expressed in the tool coordinate system.~%Orient aom~%~%# The moment of inertia of the load around the x-, y- and z-axis,~%# expressed in kgm^2.~%float32 ix~%float32 iy~%float32 iz~%~%================================================================================~%MSG: abb_rapid_msgs/Pos~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"pos\" (positions (only X, Y and Z)).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The x-, y- and z-value of the position.~%float32 x~%float32 y~%float32 z~%~%================================================================================~%MSG: abb_rapid_msgs/Orient~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"orient\" (orientation).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Quaternion 1-4.~%float32 q1~%float32 q2~%float32 q3~%float32 q4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LoadData>))
  (cl:+ 0
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'cog))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'aom))
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LoadData>))
  "Converts a ROS message object to a list"
  (cl:list 'LoadData
    (cl:cons ':mass (mass msg))
    (cl:cons ':cog (cog msg))
    (cl:cons ':aom (aom msg))
    (cl:cons ':ix (ix msg))
    (cl:cons ':iy (iy msg))
    (cl:cons ':iz (iz msg))
))
