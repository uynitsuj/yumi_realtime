; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-msg)


;//! \htmlinclude EGMSetupSettings.msg.html

(cl:defclass <EGMSetupSettings> (roslisp-msg-protocol:ros-message)
  ((use_filtering
    :reader use_filtering
    :initarg :use_filtering
    :type cl:boolean
    :initform cl:nil)
   (comm_timeout
    :reader comm_timeout
    :initarg :comm_timeout
    :type cl:float
    :initform 0.0))
)

(cl:defclass EGMSetupSettings (<EGMSetupSettings>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EGMSetupSettings>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EGMSetupSettings)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-msg:<EGMSetupSettings> is deprecated: use abb_rapid_sm_addin_msgs-msg:EGMSetupSettings instead.")))

(cl:ensure-generic-function 'use_filtering-val :lambda-list '(m))
(cl:defmethod use_filtering-val ((m <EGMSetupSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:use_filtering-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:use_filtering instead.")
  (use_filtering m))

(cl:ensure-generic-function 'comm_timeout-val :lambda-list '(m))
(cl:defmethod comm_timeout-val ((m <EGMSetupSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:comm_timeout-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:comm_timeout instead.")
  (comm_timeout m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EGMSetupSettings>) ostream)
  "Serializes a message object of type '<EGMSetupSettings>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_filtering) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'comm_timeout))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EGMSetupSettings>) istream)
  "Deserializes a message object of type '<EGMSetupSettings>"
    (cl:setf (cl:slot-value msg 'use_filtering) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'comm_timeout) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EGMSetupSettings>)))
  "Returns string type for a message object of type '<EGMSetupSettings>"
  "abb_rapid_sm_addin_msgs/EGMSetupSettings")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EGMSetupSettings)))
  "Returns string type for a message object of type 'EGMSetupSettings"
  "abb_rapid_sm_addin_msgs/EGMSetupSettings")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EGMSetupSettings>)))
  "Returns md5sum for a message object of type '<EGMSetupSettings>"
  "13bb9a91c166ef61bfdd5d16c6d9ac65")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EGMSetupSettings)))
  "Returns md5sum for a message object of type 'EGMSetupSettings"
  "13bb9a91c166ef61bfdd5d16c6d9ac65")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EGMSetupSettings>)))
  "Returns full string definition for message of type '<EGMSetupSettings>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMSetupSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMSetupSettings\" are for example used with the RAPID EGM instruction~%#   \"EGMSetupUC\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Indicator for if the EGM controller should apply filtering on the~%# EGM references.~%bool use_filtering~%~%# Communication timeout [s].~%float32 comm_timeout~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EGMSetupSettings)))
  "Returns full string definition for message of type 'EGMSetupSettings"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMSetupSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMSetupSettings\" are for example used with the RAPID EGM instruction~%#   \"EGMSetupUC\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Indicator for if the EGM controller should apply filtering on the~%# EGM references.~%bool use_filtering~%~%# Communication timeout [s].~%float32 comm_timeout~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EGMSetupSettings>))
  (cl:+ 0
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EGMSetupSettings>))
  "Converts a ROS message object to a list"
  (cl:list 'EGMSetupSettings
    (cl:cons ':use_filtering (use_filtering msg))
    (cl:cons ':comm_timeout (comm_timeout msg))
))
