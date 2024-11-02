; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-msg)


;//! \htmlinclude EGMStopSettings.msg.html

(cl:defclass <EGMStopSettings> (roslisp-msg-protocol:ros-message)
  ((ramp_out_time
    :reader ramp_out_time
    :initarg :ramp_out_time
    :type cl:float
    :initform 0.0))
)

(cl:defclass EGMStopSettings (<EGMStopSettings>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EGMStopSettings>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EGMStopSettings)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-msg:<EGMStopSettings> is deprecated: use abb_rapid_sm_addin_msgs-msg:EGMStopSettings instead.")))

(cl:ensure-generic-function 'ramp_out_time-val :lambda-list '(m))
(cl:defmethod ramp_out_time-val ((m <EGMStopSettings>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:ramp_out_time-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:ramp_out_time instead.")
  (ramp_out_time m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EGMStopSettings>) ostream)
  "Serializes a message object of type '<EGMStopSettings>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ramp_out_time))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EGMStopSettings>) istream)
  "Deserializes a message object of type '<EGMStopSettings>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ramp_out_time) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EGMStopSettings>)))
  "Returns string type for a message object of type '<EGMStopSettings>"
  "abb_rapid_sm_addin_msgs/EGMStopSettings")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EGMStopSettings)))
  "Returns string type for a message object of type 'EGMStopSettings"
  "abb_rapid_sm_addin_msgs/EGMStopSettings")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EGMStopSettings>)))
  "Returns md5sum for a message object of type '<EGMStopSettings>"
  "0e3716aa777ece435e2f7343649fe19b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EGMStopSettings)))
  "Returns md5sum for a message object of type 'EGMStopSettings"
  "0e3716aa777ece435e2f7343649fe19b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EGMStopSettings>)))
  "Returns full string definition for message of type '<EGMStopSettings>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMStopSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMStopSettings\" are for example used with the RAPID EGM instruction~%#   \"EGMStop\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Ramp out time [s].~%float32 ramp_out_time~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EGMStopSettings)))
  "Returns full string definition for message of type 'EGMStopSettings"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the custom RAPID~%#   data type \"EGMStopSettings\" from the RobotWare \"StateMachine Add-In\".~%#~%# Note:~%#   \"EGMStopSettings\" are for example used with the RAPID EGM instruction~%#   \"EGMStop\".~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Ramp out time [s].~%float32 ramp_out_time~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EGMStopSettings>))
  (cl:+ 0
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EGMStopSettings>))
  "Converts a ROS message object to a list"
  (cl:list 'EGMStopSettings
    (cl:cons ':ramp_out_time (ramp_out_time msg))
))
