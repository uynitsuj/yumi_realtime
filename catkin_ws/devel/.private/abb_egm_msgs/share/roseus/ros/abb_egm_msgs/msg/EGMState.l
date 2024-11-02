;; Auto-generated. Do not edit!


(when (boundp 'abb_egm_msgs::EGMState)
  (if (not (find-package "ABB_EGM_MSGS"))
    (make-package "ABB_EGM_MSGS"))
  (shadow 'EGMState (find-package "ABB_EGM_MSGS")))
(unless (find-package "ABB_EGM_MSGS::EGMSTATE")
  (make-package "ABB_EGM_MSGS::EGMSTATE"))

(in-package "ROS")
;;//! \htmlinclude EGMState.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass abb_egm_msgs::EGMState
  :super ros::object
  :slots (_header _egm_channels ))

(defmethod abb_egm_msgs::EGMState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:egm_channels __egm_channels) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _egm_channels __egm_channels)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:egm_channels
   (&rest __egm_channels)
   (if (keywordp (car __egm_channels))
       (send* _egm_channels __egm_channels)
     (progn
       (if __egm_channels (setq _egm_channels (car __egm_channels)))
       _egm_channels)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; abb_egm_msgs/EGMChannelState[] _egm_channels
    (apply #'+ (send-all _egm_channels :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; abb_egm_msgs/EGMChannelState[] _egm_channels
     (write-long (length _egm_channels) s)
     (dolist (elem _egm_channels)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; abb_egm_msgs/EGMChannelState[] _egm_channels
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _egm_channels (let (r) (dotimes (i n) (push (instance abb_egm_msgs::EGMChannelState :init) r)) r))
     (dolist (elem- _egm_channels)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get abb_egm_msgs::EGMState :md5sum-) "a239c0f02980c16cd34f9e56fbc03ff7")
(setf (get abb_egm_msgs::EGMState :datatype-) "abb_egm_msgs/EGMState")
(setf (get abb_egm_msgs::EGMState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of all (known) EGM communication channels.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Metadata.
std_msgs/Header header

# EGM channel states.
EGMChannelState[] egm_channels

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: abb_egm_msgs/EGMChannelState
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of an EGM communication channel. I.e. a channel between an EGM client,
#   in an ABB robot controller system, and an external EGM server.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# EGM client states:
uint8 EGM_UNDEFINED = 1
uint8 EGM_ERROR     = 2
uint8 EGM_STOPPED   = 3
uint8 EGM_RUNNING   = 4

# Motor states:
uint8 MOTORS_UNDEFINED = 1
uint8 MOTORS_ON        = 2
uint8 MOTORS_OFF       = 3

# RAPID states:
uint8 RAPID_UNDEFINED = 1
uint8 RAPID_STOPPED   = 2
uint8 RAPID_RUNNING   = 3

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The EGM channel's name.
string name

# The EGM channel's active/inactive status.
bool active

# Indicator for if the EGM convergence criteria is currently fulfilled or not.
bool egm_convergence_met

# EGM client state.
uint8 egm_client_state

# Motor state.
uint8 motor_state

# RAPID execution state.
uint8 rapid_execution_state

# Utilization rate of the connected mechanical unit.
#
# An utilization rate > 100 mean that EGM motion references are
# too aggressive and should be reduced.
float64 utilization_rate

")



(provide :abb_egm_msgs/EGMState "a239c0f02980c16cd34f9e56fbc03ff7")


