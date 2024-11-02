;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::EGMSettings)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'EGMSettings (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::EGMSETTINGS")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::EGMSETTINGS"))

(in-package "ROS")
;;//! \htmlinclude EGMSettings.msg.html


(defclass abb_rapid_sm_addin_msgs::EGMSettings
  :super ros::object
  :slots (_allow_egm_motions _use_presync _setup_uc _activate _run _stop ))

(defmethod abb_rapid_sm_addin_msgs::EGMSettings
  (:init
   (&key
    ((:allow_egm_motions __allow_egm_motions) nil)
    ((:use_presync __use_presync) nil)
    ((:setup_uc __setup_uc) (instance abb_rapid_sm_addin_msgs::EGMSetupSettings :init))
    ((:activate __activate) (instance abb_rapid_sm_addin_msgs::EGMActivateSettings :init))
    ((:run __run) (instance abb_rapid_sm_addin_msgs::EGMRunSettings :init))
    ((:stop __stop) (instance abb_rapid_sm_addin_msgs::EGMStopSettings :init))
    )
   (send-super :init)
   (setq _allow_egm_motions __allow_egm_motions)
   (setq _use_presync __use_presync)
   (setq _setup_uc __setup_uc)
   (setq _activate __activate)
   (setq _run __run)
   (setq _stop __stop)
   self)
  (:allow_egm_motions
   (&optional (__allow_egm_motions :null))
   (if (not (eq __allow_egm_motions :null)) (setq _allow_egm_motions __allow_egm_motions)) _allow_egm_motions)
  (:use_presync
   (&optional (__use_presync :null))
   (if (not (eq __use_presync :null)) (setq _use_presync __use_presync)) _use_presync)
  (:setup_uc
   (&rest __setup_uc)
   (if (keywordp (car __setup_uc))
       (send* _setup_uc __setup_uc)
     (progn
       (if __setup_uc (setq _setup_uc (car __setup_uc)))
       _setup_uc)))
  (:activate
   (&rest __activate)
   (if (keywordp (car __activate))
       (send* _activate __activate)
     (progn
       (if __activate (setq _activate (car __activate)))
       _activate)))
  (:run
   (&rest __run)
   (if (keywordp (car __run))
       (send* _run __run)
     (progn
       (if __run (setq _run (car __run)))
       _run)))
  (:stop
   (&rest __stop)
   (if (keywordp (car __stop))
       (send* _stop __stop)
     (progn
       (if __stop (setq _stop (car __stop)))
       _stop)))
  (:serialization-length
   ()
   (+
    ;; bool _allow_egm_motions
    1
    ;; bool _use_presync
    1
    ;; abb_rapid_sm_addin_msgs/EGMSetupSettings _setup_uc
    (send _setup_uc :serialization-length)
    ;; abb_rapid_sm_addin_msgs/EGMActivateSettings _activate
    (send _activate :serialization-length)
    ;; abb_rapid_sm_addin_msgs/EGMRunSettings _run
    (send _run :serialization-length)
    ;; abb_rapid_sm_addin_msgs/EGMStopSettings _stop
    (send _stop :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _allow_egm_motions
       (if _allow_egm_motions (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_presync
       (if _use_presync (write-byte -1 s) (write-byte 0 s))
     ;; abb_rapid_sm_addin_msgs/EGMSetupSettings _setup_uc
       (send _setup_uc :serialize s)
     ;; abb_rapid_sm_addin_msgs/EGMActivateSettings _activate
       (send _activate :serialize s)
     ;; abb_rapid_sm_addin_msgs/EGMRunSettings _run
       (send _run :serialize s)
     ;; abb_rapid_sm_addin_msgs/EGMStopSettings _stop
       (send _stop :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _allow_egm_motions
     (setq _allow_egm_motions (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_presync
     (setq _use_presync (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; abb_rapid_sm_addin_msgs/EGMSetupSettings _setup_uc
     (send _setup_uc :deserialize buf ptr-) (incf ptr- (send _setup_uc :serialization-length))
   ;; abb_rapid_sm_addin_msgs/EGMActivateSettings _activate
     (send _activate :deserialize buf ptr-) (incf ptr- (send _activate :serialization-length))
   ;; abb_rapid_sm_addin_msgs/EGMRunSettings _run
     (send _run :deserialize buf ptr-) (incf ptr- (send _run :serialization-length))
   ;; abb_rapid_sm_addin_msgs/EGMStopSettings _stop
     (send _stop :deserialize buf ptr-) (incf ptr- (send _stop :serialization-length))
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::EGMSettings :md5sum-) "7c458fd979a6a372744a25dc9e88085d")
(setf (get abb_rapid_sm_addin_msgs::EGMSettings :datatype-) "abb_rapid_sm_addin_msgs/EGMSettings")
(setf (get abb_rapid_sm_addin_msgs::EGMSettings :definition-)
      "#-------------------------------------------------------------------------------
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

")



(provide :abb_rapid_sm_addin_msgs/EGMSettings "7c458fd979a6a372744a25dc9e88085d")


