;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::EGMRunSettings)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'EGMRunSettings (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::EGMRUNSETTINGS")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::EGMRUNSETTINGS"))

(in-package "ROS")
;;//! \htmlinclude EGMRunSettings.msg.html
(if (not (find-package "ABB_RAPID_MSGS"))
  (ros::roseus-add-msgs "abb_rapid_msgs"))


(defclass abb_rapid_sm_addin_msgs::EGMRunSettings
  :super ros::object
  :slots (_cond_time _ramp_in_time _offset _pos_corr_gain ))

(defmethod abb_rapid_sm_addin_msgs::EGMRunSettings
  (:init
   (&key
    ((:cond_time __cond_time) 0.0)
    ((:ramp_in_time __ramp_in_time) 0.0)
    ((:offset __offset) (instance abb_rapid_msgs::Pose :init))
    ((:pos_corr_gain __pos_corr_gain) 0.0)
    )
   (send-super :init)
   (setq _cond_time (float __cond_time))
   (setq _ramp_in_time (float __ramp_in_time))
   (setq _offset __offset)
   (setq _pos_corr_gain (float __pos_corr_gain))
   self)
  (:cond_time
   (&optional __cond_time)
   (if __cond_time (setq _cond_time __cond_time)) _cond_time)
  (:ramp_in_time
   (&optional __ramp_in_time)
   (if __ramp_in_time (setq _ramp_in_time __ramp_in_time)) _ramp_in_time)
  (:offset
   (&rest __offset)
   (if (keywordp (car __offset))
       (send* _offset __offset)
     (progn
       (if __offset (setq _offset (car __offset)))
       _offset)))
  (:pos_corr_gain
   (&optional __pos_corr_gain)
   (if __pos_corr_gain (setq _pos_corr_gain __pos_corr_gain)) _pos_corr_gain)
  (:serialization-length
   ()
   (+
    ;; float32 _cond_time
    4
    ;; float32 _ramp_in_time
    4
    ;; abb_rapid_msgs/Pose _offset
    (send _offset :serialization-length)
    ;; float32 _pos_corr_gain
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _cond_time
       (sys::poke _cond_time (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _ramp_in_time
       (sys::poke _ramp_in_time (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; abb_rapid_msgs/Pose _offset
       (send _offset :serialize s)
     ;; float32 _pos_corr_gain
       (sys::poke _pos_corr_gain (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _cond_time
     (setq _cond_time (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _ramp_in_time
     (setq _ramp_in_time (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; abb_rapid_msgs/Pose _offset
     (send _offset :deserialize buf ptr-) (incf ptr- (send _offset :serialization-length))
   ;; float32 _pos_corr_gain
     (setq _pos_corr_gain (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::EGMRunSettings :md5sum-) "f6a2d3b6d7d6ffc880c4bc86612a6691")
(setf (get abb_rapid_sm_addin_msgs::EGMRunSettings :datatype-) "abb_rapid_sm_addin_msgs/EGMRunSettings")
(setf (get abb_rapid_sm_addin_msgs::EGMRunSettings :definition-)
      "#-------------------------------------------------------------------------------
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

")



(provide :abb_rapid_sm_addin_msgs/EGMRunSettings "f6a2d3b6d7d6ffc880c4bc86612a6691")


