;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::SetEGMSettings)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'SetEGMSettings (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETEGMSETTINGS")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETEGMSETTINGS"))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETEGMSETTINGSREQUEST")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETEGMSETTINGSREQUEST"))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETEGMSETTINGSRESPONSE")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETEGMSETTINGSRESPONSE"))

(in-package "ROS")





(defclass abb_rapid_sm_addin_msgs::SetEGMSettingsRequest
  :super ros::object
  :slots (_task _settings ))

(defmethod abb_rapid_sm_addin_msgs::SetEGMSettingsRequest
  (:init
   (&key
    ((:task __task) "")
    ((:settings __settings) (instance abb_rapid_sm_addin_msgs::EGMSettings :init))
    )
   (send-super :init)
   (setq _task (string __task))
   (setq _settings __settings)
   self)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:settings
   (&rest __settings)
   (if (keywordp (car __settings))
       (send* _settings __settings)
     (progn
       (if __settings (setq _settings (car __settings)))
       _settings)))
  (:serialization-length
   ()
   (+
    ;; string _task
    4 (length _task)
    ;; abb_rapid_sm_addin_msgs/EGMSettings _settings
    (send _settings :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; abb_rapid_sm_addin_msgs/EGMSettings _settings
       (send _settings :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; abb_rapid_sm_addin_msgs/EGMSettings _settings
     (send _settings :deserialize buf ptr-) (incf ptr- (send _settings :serialization-length))
   ;;
   self)
  )

(defclass abb_rapid_sm_addin_msgs::SetEGMSettingsResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_rapid_sm_addin_msgs::SetEGMSettingsResponse
  (:init
   (&key
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_rapid_sm_addin_msgs::SetEGMSettings
  :super ros::object
  :slots ())

(setf (get abb_rapid_sm_addin_msgs::SetEGMSettings :md5sum-) "1308f55a0badf1657b70daea7afc5401")
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettings :datatype-) "abb_rapid_sm_addin_msgs/SetEGMSettings")
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettings :request) abb_rapid_sm_addin_msgs::SetEGMSettingsRequest)
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettings :response) abb_rapid_sm_addin_msgs::SetEGMSettingsResponse)

(defmethod abb_rapid_sm_addin_msgs::SetEGMSettingsRequest
  (:response () (instance abb_rapid_sm_addin_msgs::SetEGMSettingsResponse :init)))

(setf (get abb_rapid_sm_addin_msgs::SetEGMSettingsRequest :md5sum-) "1308f55a0badf1657b70daea7afc5401")
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettingsRequest :datatype-) "abb_rapid_sm_addin_msgs/SetEGMSettingsRequest")
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettingsRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a predefined RAPID variable of the custom RAPID data type \"EGMSettings\".
#
# Note:
#   This variable is defined in each RAPID motion task, in ABB robot controller
#   systems, that are using the RobotWare \"StateMachine Add-In\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# The RAPID task, where the variable exists.
string task

# New settings to use for RAPID EGM instructions.
EGMSettings settings


================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMSettings\" from the RobotWare \"StateMachine Add-In\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Indicator for if EGM motions are allowed to start.
bool allow_egm_motions

# Indicator for if RAPID tasks should be synced before starting EGM motions
# (only used in multi robot systems).
bool use_presync

# Settings for EGMSetupUC instructions.
EGMSetupSettings setup_uc

# Settings for EGMAct instructions.
EGMActivateSettings activate

# Settings for EGMRun instructions.
EGMRunSettings run

# Settings for EGMStop instructions.
EGMStopSettings stop

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMSetupSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMSetupSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMSetupSettings\" are for example used with the RAPID EGM instruction
#   \"EGMSetupUC\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Indicator for if the EGM controller should apply filtering on the
# EGM references.
bool use_filtering

# Communication timeout [s].
float32 comm_timeout

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMActivateSettings
#-------------------------------------------------------------------------------
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

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMRunSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMRunSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMRunSettings\" are for example used with the RAPID EGM instructions
#   \"EGMRunJoint\" and \"EGMRunPose\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Condition time [s] for how long an EGM motion's convergence criteria must be
# fulfilled before the motion is considered to be finished.
float32 cond_time

# Ramp in time [s].
float32 ramp_in_time

# Static offset applied on top of the EGM references
# (only used in EGM pose mode).
abb_rapid_msgs/Pose offset

# Position correction gain for the EGM controller.
float32 pos_corr_gain

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMStopSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMStopSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMStopSettings\" are for example used with the RAPID EGM instruction
#   \"EGMStop\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Ramp out time [s].
float32 ramp_out_time
---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_rapid_sm_addin_msgs::SetEGMSettingsResponse :md5sum-) "1308f55a0badf1657b70daea7afc5401")
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettingsResponse :datatype-) "abb_rapid_sm_addin_msgs/SetEGMSettingsResponse")
(setf (get abb_rapid_sm_addin_msgs::SetEGMSettingsResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a predefined RAPID variable of the custom RAPID data type \"EGMSettings\".
#
# Note:
#   This variable is defined in each RAPID motion task, in ABB robot controller
#   systems, that are using the RobotWare \"StateMachine Add-In\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# The RAPID task, where the variable exists.
string task

# New settings to use for RAPID EGM instructions.
EGMSettings settings


================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMSettings\" from the RobotWare \"StateMachine Add-In\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Indicator for if EGM motions are allowed to start.
bool allow_egm_motions

# Indicator for if RAPID tasks should be synced before starting EGM motions
# (only used in multi robot systems).
bool use_presync

# Settings for EGMSetupUC instructions.
EGMSetupSettings setup_uc

# Settings for EGMAct instructions.
EGMActivateSettings activate

# Settings for EGMRun instructions.
EGMRunSettings run

# Settings for EGMStop instructions.
EGMStopSettings stop

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMSetupSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMSetupSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMSetupSettings\" are for example used with the RAPID EGM instruction
#   \"EGMSetupUC\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Indicator for if the EGM controller should apply filtering on the
# EGM references.
bool use_filtering

# Communication timeout [s].
float32 comm_timeout

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMActivateSettings
#-------------------------------------------------------------------------------
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

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMRunSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMRunSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMRunSettings\" are for example used with the RAPID EGM instructions
#   \"EGMRunJoint\" and \"EGMRunPose\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Condition time [s] for how long an EGM motion's convergence criteria must be
# fulfilled before the motion is considered to be finished.
float32 cond_time

# Ramp in time [s].
float32 ramp_in_time

# Static offset applied on top of the EGM references
# (only used in EGM pose mode).
abb_rapid_msgs/Pose offset

# Position correction gain for the EGM controller.
float32 pos_corr_gain

================================================================================
MSG: abb_rapid_sm_addin_msgs/EGMStopSettings
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the custom RAPID
#   data type \"EGMStopSettings\" from the RobotWare \"StateMachine Add-In\".
#
# Note:
#   \"EGMStopSettings\" are for example used with the RAPID EGM instruction
#   \"EGMStop\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Ramp out time [s].
float32 ramp_out_time
---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_rapid_sm_addin_msgs/SetEGMSettings "1308f55a0badf1657b70daea7afc5401")


