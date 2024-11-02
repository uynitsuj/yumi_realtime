; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-msg)


;//! \htmlinclude SystemState.msg.html

(cl:defclass <SystemState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (motors_on
    :reader motors_on
    :initarg :motors_on
    :type cl:boolean
    :initform cl:nil)
   (auto_mode
    :reader auto_mode
    :initarg :auto_mode
    :type cl:boolean
    :initform cl:nil)
   (rapid_running
    :reader rapid_running
    :initarg :rapid_running
    :type cl:boolean
    :initform cl:nil)
   (rapid_tasks
    :reader rapid_tasks
    :initarg :rapid_tasks
    :type (cl:vector abb_robot_msgs-msg:RAPIDTaskState)
   :initform (cl:make-array 0 :element-type 'abb_robot_msgs-msg:RAPIDTaskState :initial-element (cl:make-instance 'abb_robot_msgs-msg:RAPIDTaskState)))
   (mechanical_units
    :reader mechanical_units
    :initarg :mechanical_units
    :type (cl:vector abb_robot_msgs-msg:MechanicalUnitState)
   :initform (cl:make-array 0 :element-type 'abb_robot_msgs-msg:MechanicalUnitState :initial-element (cl:make-instance 'abb_robot_msgs-msg:MechanicalUnitState))))
)

(cl:defclass SystemState (<SystemState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SystemState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SystemState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-msg:<SystemState> is deprecated: use abb_robot_msgs-msg:SystemState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <SystemState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:header-val is deprecated.  Use abb_robot_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'motors_on-val :lambda-list '(m))
(cl:defmethod motors_on-val ((m <SystemState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:motors_on-val is deprecated.  Use abb_robot_msgs-msg:motors_on instead.")
  (motors_on m))

(cl:ensure-generic-function 'auto_mode-val :lambda-list '(m))
(cl:defmethod auto_mode-val ((m <SystemState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:auto_mode-val is deprecated.  Use abb_robot_msgs-msg:auto_mode instead.")
  (auto_mode m))

(cl:ensure-generic-function 'rapid_running-val :lambda-list '(m))
(cl:defmethod rapid_running-val ((m <SystemState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:rapid_running-val is deprecated.  Use abb_robot_msgs-msg:rapid_running instead.")
  (rapid_running m))

(cl:ensure-generic-function 'rapid_tasks-val :lambda-list '(m))
(cl:defmethod rapid_tasks-val ((m <SystemState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:rapid_tasks-val is deprecated.  Use abb_robot_msgs-msg:rapid_tasks instead.")
  (rapid_tasks m))

(cl:ensure-generic-function 'mechanical_units-val :lambda-list '(m))
(cl:defmethod mechanical_units-val ((m <SystemState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-msg:mechanical_units-val is deprecated.  Use abb_robot_msgs-msg:mechanical_units instead.")
  (mechanical_units m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SystemState>) ostream)
  "Serializes a message object of type '<SystemState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'motors_on) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'auto_mode) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'rapid_running) 1 0)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'rapid_tasks))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'rapid_tasks))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'mechanical_units))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'mechanical_units))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SystemState>) istream)
  "Deserializes a message object of type '<SystemState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:slot-value msg 'motors_on) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'auto_mode) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'rapid_running) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'rapid_tasks) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'rapid_tasks)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'abb_robot_msgs-msg:RAPIDTaskState))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'mechanical_units) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'mechanical_units)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'abb_robot_msgs-msg:MechanicalUnitState))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SystemState>)))
  "Returns string type for a message object of type '<SystemState>"
  "abb_robot_msgs/SystemState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SystemState)))
  "Returns string type for a message object of type 'SystemState"
  "abb_robot_msgs/SystemState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SystemState>)))
  "Returns md5sum for a message object of type '<SystemState>"
  "46309669ed6ba4863375b03174d00ad7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SystemState)))
  "Returns md5sum for a message object of type 'SystemState"
  "46309669ed6ba4863375b03174d00ad7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SystemState>)))
  "Returns full string definition for message of type '<SystemState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Metadata.~%std_msgs/Header header~%~%# Motors on/off indicator.~%bool motors_on~%~%# Auto/manual mode indicator.~%bool auto_mode~%~%# RAPID running/stopped indicator.~%bool rapid_running~%~%# RAPID task states.~%RAPIDTaskState[] rapid_tasks~%~%# Mechanical unit states.~%MechanicalUnitState[] mechanical_units~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDTaskState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a RAPID task defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# RAPID execution states:~%uint8 EXECUTION_STATE_UNKNOWN       = 1~%uint8 EXECUTION_STATE_READY         = 2~%uint8 EXECUTION_STATE_STOPPED       = 3~%uint8 EXECUTION_STATE_STARTED       = 4~%uint8 EXECUTION_STATE_UNINITIALIZED = 5~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task's name.~%string name~%~%# The RAPID task's activated/deactivated status.~%bool activated~%~%# The RAPID task's execution state.~%uint8 execution_state~%~%# The RAPID task is a motion task.~%bool motion_task~%~%================================================================================~%MSG: abb_robot_msgs/MechanicalUnitState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a mechanical unit defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mechanical unit's name.~%string name~%~%# The mechanical unit's activated/deactivated status.~%bool activated~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SystemState)))
  "Returns full string definition for message of type 'SystemState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Metadata.~%std_msgs/Header header~%~%# Motors on/off indicator.~%bool motors_on~%~%# Auto/manual mode indicator.~%bool auto_mode~%~%# RAPID running/stopped indicator.~%bool rapid_running~%~%# RAPID task states.~%RAPIDTaskState[] rapid_tasks~%~%# Mechanical unit states.~%MechanicalUnitState[] mechanical_units~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: abb_robot_msgs/RAPIDTaskState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a RAPID task defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# RAPID execution states:~%uint8 EXECUTION_STATE_UNKNOWN       = 1~%uint8 EXECUTION_STATE_READY         = 2~%uint8 EXECUTION_STATE_STOPPED       = 3~%uint8 EXECUTION_STATE_STARTED       = 4~%uint8 EXECUTION_STATE_UNINITIALIZED = 5~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task's name.~%string name~%~%# The RAPID task's activated/deactivated status.~%bool activated~%~%# The RAPID task's execution state.~%uint8 execution_state~%~%# The RAPID task is a motion task.~%bool motion_task~%~%================================================================================~%MSG: abb_robot_msgs/MechanicalUnitState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of a mechanical unit defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The mechanical unit's name.~%string name~%~%# The mechanical unit's activated/deactivated status.~%bool activated~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SystemState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     1
     1
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'rapid_tasks) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'mechanical_units) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SystemState>))
  "Converts a ROS message object to a list"
  (cl:list 'SystemState
    (cl:cons ':header (header msg))
    (cl:cons ':motors_on (motors_on msg))
    (cl:cons ':auto_mode (auto_mode msg))
    (cl:cons ':rapid_running (rapid_running msg))
    (cl:cons ':rapid_tasks (rapid_tasks msg))
    (cl:cons ':mechanical_units (mechanical_units msg))
))
