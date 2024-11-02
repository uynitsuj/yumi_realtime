;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SystemState)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SystemState (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SYSTEMSTATE")
  (make-package "ABB_ROBOT_MSGS::SYSTEMSTATE"))

(in-package "ROS")
;;//! \htmlinclude SystemState.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass abb_robot_msgs::SystemState
  :super ros::object
  :slots (_header _motors_on _auto_mode _rapid_running _rapid_tasks _mechanical_units ))

(defmethod abb_robot_msgs::SystemState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:motors_on __motors_on) nil)
    ((:auto_mode __auto_mode) nil)
    ((:rapid_running __rapid_running) nil)
    ((:rapid_tasks __rapid_tasks) ())
    ((:mechanical_units __mechanical_units) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _motors_on __motors_on)
   (setq _auto_mode __auto_mode)
   (setq _rapid_running __rapid_running)
   (setq _rapid_tasks __rapid_tasks)
   (setq _mechanical_units __mechanical_units)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:motors_on
   (&optional (__motors_on :null))
   (if (not (eq __motors_on :null)) (setq _motors_on __motors_on)) _motors_on)
  (:auto_mode
   (&optional (__auto_mode :null))
   (if (not (eq __auto_mode :null)) (setq _auto_mode __auto_mode)) _auto_mode)
  (:rapid_running
   (&optional (__rapid_running :null))
   (if (not (eq __rapid_running :null)) (setq _rapid_running __rapid_running)) _rapid_running)
  (:rapid_tasks
   (&rest __rapid_tasks)
   (if (keywordp (car __rapid_tasks))
       (send* _rapid_tasks __rapid_tasks)
     (progn
       (if __rapid_tasks (setq _rapid_tasks (car __rapid_tasks)))
       _rapid_tasks)))
  (:mechanical_units
   (&rest __mechanical_units)
   (if (keywordp (car __mechanical_units))
       (send* _mechanical_units __mechanical_units)
     (progn
       (if __mechanical_units (setq _mechanical_units (car __mechanical_units)))
       _mechanical_units)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; bool _motors_on
    1
    ;; bool _auto_mode
    1
    ;; bool _rapid_running
    1
    ;; abb_robot_msgs/RAPIDTaskState[] _rapid_tasks
    (apply #'+ (send-all _rapid_tasks :serialization-length)) 4
    ;; abb_robot_msgs/MechanicalUnitState[] _mechanical_units
    (apply #'+ (send-all _mechanical_units :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; bool _motors_on
       (if _motors_on (write-byte -1 s) (write-byte 0 s))
     ;; bool _auto_mode
       (if _auto_mode (write-byte -1 s) (write-byte 0 s))
     ;; bool _rapid_running
       (if _rapid_running (write-byte -1 s) (write-byte 0 s))
     ;; abb_robot_msgs/RAPIDTaskState[] _rapid_tasks
     (write-long (length _rapid_tasks) s)
     (dolist (elem _rapid_tasks)
       (send elem :serialize s)
       )
     ;; abb_robot_msgs/MechanicalUnitState[] _mechanical_units
     (write-long (length _mechanical_units) s)
     (dolist (elem _mechanical_units)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; bool _motors_on
     (setq _motors_on (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _auto_mode
     (setq _auto_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _rapid_running
     (setq _rapid_running (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; abb_robot_msgs/RAPIDTaskState[] _rapid_tasks
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _rapid_tasks (let (r) (dotimes (i n) (push (instance abb_robot_msgs::RAPIDTaskState :init) r)) r))
     (dolist (elem- _rapid_tasks)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; abb_robot_msgs/MechanicalUnitState[] _mechanical_units
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _mechanical_units (let (r) (dotimes (i n) (push (instance abb_robot_msgs::MechanicalUnitState :init) r)) r))
     (dolist (elem- _mechanical_units)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get abb_robot_msgs::SystemState :md5sum-) "46309669ed6ba4863375b03174d00ad7")
(setf (get abb_robot_msgs::SystemState :datatype-) "abb_robot_msgs/SystemState")
(setf (get abb_robot_msgs::SystemState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Metadata.
std_msgs/Header header

# Motors on/off indicator.
bool motors_on

# Auto/manual mode indicator.
bool auto_mode

# RAPID running/stopped indicator.
bool rapid_running

# RAPID task states.
RAPIDTaskState[] rapid_tasks

# Mechanical unit states.
MechanicalUnitState[] mechanical_units

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
MSG: abb_robot_msgs/RAPIDTaskState
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of a RAPID task defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# RAPID execution states:
uint8 EXECUTION_STATE_UNKNOWN       = 1
uint8 EXECUTION_STATE_READY         = 2
uint8 EXECUTION_STATE_STOPPED       = 3
uint8 EXECUTION_STATE_STARTED       = 4
uint8 EXECUTION_STATE_UNINITIALIZED = 5

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The RAPID task's name.
string name

# The RAPID task's activated/deactivated status.
bool activated

# The RAPID task's execution state.
uint8 execution_state

# The RAPID task is a motion task.
bool motion_task

================================================================================
MSG: abb_robot_msgs/MechanicalUnitState
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of a mechanical unit defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The mechanical unit's name.
string name

# The mechanical unit's activated/deactivated status.
bool activated

")



(provide :abb_robot_msgs/SystemState "46309669ed6ba4863375b03174d00ad7")


