;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::EGMActivateSettings)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'EGMActivateSettings (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::EGMACTIVATESETTINGS")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::EGMACTIVATESETTINGS"))

(in-package "ROS")
;;//! \htmlinclude EGMActivateSettings.msg.html
(if (not (find-package "ABB_RAPID_MSGS"))
  (ros::roseus-add-msgs "abb_rapid_msgs"))


(defclass abb_rapid_sm_addin_msgs::EGMActivateSettings
  :super ros::object
  :slots (_tool _wobj _correction_frame _sensor_frame _cond_min_max _lp_filter _sample_rate _max_speed_deviation ))

(defmethod abb_rapid_sm_addin_msgs::EGMActivateSettings
  (:init
   (&key
    ((:tool __tool) (instance abb_rapid_msgs::ToolData :init))
    ((:wobj __wobj) (instance abb_rapid_msgs::WObjData :init))
    ((:correction_frame __correction_frame) (instance abb_rapid_msgs::Pose :init))
    ((:sensor_frame __sensor_frame) (instance abb_rapid_msgs::Pose :init))
    ((:cond_min_max __cond_min_max) 0.0)
    ((:lp_filter __lp_filter) 0.0)
    ((:sample_rate __sample_rate) 0)
    ((:max_speed_deviation __max_speed_deviation) 0.0)
    )
   (send-super :init)
   (setq _tool __tool)
   (setq _wobj __wobj)
   (setq _correction_frame __correction_frame)
   (setq _sensor_frame __sensor_frame)
   (setq _cond_min_max (float __cond_min_max))
   (setq _lp_filter (float __lp_filter))
   (setq _sample_rate (round __sample_rate))
   (setq _max_speed_deviation (float __max_speed_deviation))
   self)
  (:tool
   (&rest __tool)
   (if (keywordp (car __tool))
       (send* _tool __tool)
     (progn
       (if __tool (setq _tool (car __tool)))
       _tool)))
  (:wobj
   (&rest __wobj)
   (if (keywordp (car __wobj))
       (send* _wobj __wobj)
     (progn
       (if __wobj (setq _wobj (car __wobj)))
       _wobj)))
  (:correction_frame
   (&rest __correction_frame)
   (if (keywordp (car __correction_frame))
       (send* _correction_frame __correction_frame)
     (progn
       (if __correction_frame (setq _correction_frame (car __correction_frame)))
       _correction_frame)))
  (:sensor_frame
   (&rest __sensor_frame)
   (if (keywordp (car __sensor_frame))
       (send* _sensor_frame __sensor_frame)
     (progn
       (if __sensor_frame (setq _sensor_frame (car __sensor_frame)))
       _sensor_frame)))
  (:cond_min_max
   (&optional __cond_min_max)
   (if __cond_min_max (setq _cond_min_max __cond_min_max)) _cond_min_max)
  (:lp_filter
   (&optional __lp_filter)
   (if __lp_filter (setq _lp_filter __lp_filter)) _lp_filter)
  (:sample_rate
   (&optional __sample_rate)
   (if __sample_rate (setq _sample_rate __sample_rate)) _sample_rate)
  (:max_speed_deviation
   (&optional __max_speed_deviation)
   (if __max_speed_deviation (setq _max_speed_deviation __max_speed_deviation)) _max_speed_deviation)
  (:serialization-length
   ()
   (+
    ;; abb_rapid_msgs/ToolData _tool
    (send _tool :serialization-length)
    ;; abb_rapid_msgs/WObjData _wobj
    (send _wobj :serialization-length)
    ;; abb_rapid_msgs/Pose _correction_frame
    (send _correction_frame :serialization-length)
    ;; abb_rapid_msgs/Pose _sensor_frame
    (send _sensor_frame :serialization-length)
    ;; float32 _cond_min_max
    4
    ;; float32 _lp_filter
    4
    ;; uint8 _sample_rate
    1
    ;; float32 _max_speed_deviation
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_rapid_msgs/ToolData _tool
       (send _tool :serialize s)
     ;; abb_rapid_msgs/WObjData _wobj
       (send _wobj :serialize s)
     ;; abb_rapid_msgs/Pose _correction_frame
       (send _correction_frame :serialize s)
     ;; abb_rapid_msgs/Pose _sensor_frame
       (send _sensor_frame :serialize s)
     ;; float32 _cond_min_max
       (sys::poke _cond_min_max (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _lp_filter
       (sys::poke _lp_filter (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint8 _sample_rate
       (write-byte _sample_rate s)
     ;; float32 _max_speed_deviation
       (sys::poke _max_speed_deviation (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_rapid_msgs/ToolData _tool
     (send _tool :deserialize buf ptr-) (incf ptr- (send _tool :serialization-length))
   ;; abb_rapid_msgs/WObjData _wobj
     (send _wobj :deserialize buf ptr-) (incf ptr- (send _wobj :serialization-length))
   ;; abb_rapid_msgs/Pose _correction_frame
     (send _correction_frame :deserialize buf ptr-) (incf ptr- (send _correction_frame :serialization-length))
   ;; abb_rapid_msgs/Pose _sensor_frame
     (send _sensor_frame :deserialize buf ptr-) (incf ptr- (send _sensor_frame :serialization-length))
   ;; float32 _cond_min_max
     (setq _cond_min_max (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _lp_filter
     (setq _lp_filter (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint8 _sample_rate
     (setq _sample_rate (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _max_speed_deviation
     (setq _max_speed_deviation (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::EGMActivateSettings :md5sum-) "edc93be8ce076e9147c54b4dc80e7974")
(setf (get abb_rapid_sm_addin_msgs::EGMActivateSettings :datatype-) "abb_rapid_sm_addin_msgs/EGMActivateSettings")
(setf (get abb_rapid_sm_addin_msgs::EGMActivateSettings :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMActivateSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMActivateSettings\" are for example used with the RAPID EGM instructions
#   \"EGMActJoint\" and \"EGMActPose\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Tool to use for EGM motions.
abb_rapid_msgs/ToolData tool

# Work object to use for EGM motions.
abb_rapid_msgs/WObjData wobj

# Correction frame to use for EGM motions (only used in EGM pose mode).
abb_rapid_msgs/Pose correction_frame

# Sensor frame to use for EGM motions (only used in EGM pose mode).
abb_rapid_msgs/Pose sensor_frame

# Condition threshold value [deg or mm] for when an EGM motion's convergence
# criteria is considered to be fulfilled.
float32 cond_min_max

# Low pass filter bandwidth [Hz] for the EGM controller.
float32 lp_filter

# Sample rate [ms] for the EGM communication.
uint8 sample_rate

# Maximum admitted joint speed [deg/s].
float32 max_speed_deviation

================================================================================
MSG: abb_rapid_msgs/ToolData
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"tooldata\" (tool data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Defines whether or not the robot is holding the tool:
# - TRUE: The robot is holding the tool.
# - FALSE: The robot is not holding the tool, that is, a stationary tool.
bool robhold

# The tool coordinate system, that is:
# - The position of the TCP (x, y and z) in mm,
#   expressed in the wrist coordinate system (tool0).
# - The orientation of the tool coordinate system,
#   expressed in the wrist coordinate system.
Pose tframe

# The load of the tool (if robot held tool).
# The load of the gripper holding the work object (if stationary tool).
LoadData tload

================================================================================
MSG: abb_rapid_msgs/Pose
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"pose\" (coordinate transformations).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The displacement in position (x, y, and z) of the coordinate system.
Pos trans

# The rotation of the coordinate system.
Orient rot

================================================================================
MSG: abb_rapid_msgs/Pos
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"pos\" (positions (only X, Y and Z)).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The x-, y- and z-value of the position.
float32 x
float32 y
float32 z

================================================================================
MSG: abb_rapid_msgs/Orient
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"orient\" (orientation).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Quaternion 1-4.
float32 q1
float32 q2
float32 q3
float32 q4

================================================================================
MSG: abb_rapid_msgs/LoadData
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"loaddata\" (load data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The mass (weight) of the load in kg.
float32 mass

# The center of gravity of the payload, expressed in mm,
# in the tool coordinate system if the robot is holding the tool.
#
# If a stationary tool is used then the center of gravity for the payload
# held by the gripper is expressed in the object frame of the work object
# coordinate system moved by the robot.
Pos cog

# The orientation of the axes of moment.
#
# These are the principal axes of the payload
# moment of inertia with origin in cog.
#
# If the robot is holding the tool, the axes of moment
# are expressed in the tool coordinate system.
Orient aom

# The moment of inertia of the load around the x-, y- and z-axis,
# expressed in kgm^2.
float32 ix
float32 iy
float32 iz

================================================================================
MSG: abb_rapid_msgs/WObjData
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"wobjdata\" (work object data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Defines whether or not the robot in the actual
# program task is holding the work object:
# - TRUE: The robot is holding the work object,
#         i.e. using a stationary tool.
# - FALSE: The robot is not holding the work object,
#          i.e. the robot is holding the tool.
bool robhold

# Defines whether or not a fixed user coordinate system is used:
# - TRUE: Fixed user coordinate system.
# - FALSE: Movable user coordinate system,
#          i.e. coordinated external axes are used.
#          Also to be used in a MultiMove system in
#          semicoordinated or synchronized coordinated mode.
bool ufprog

# The mechanical unit with which the robot movements are coordinated.
#
# Only specified in the case of movable user
# coordinate systems (ufprog is FALSE).
string ufmec

# The user coordinate system, i.e. the position
# of the current work surface or fixture.
Pose uframe

# The object coordinate system, i.e. the position
# of the current work object.
Pose oframe

")



(provide :abb_rapid_sm_addin_msgs/EGMActivateSettings "edc93be8ce076e9147c54b4dc80e7974")


