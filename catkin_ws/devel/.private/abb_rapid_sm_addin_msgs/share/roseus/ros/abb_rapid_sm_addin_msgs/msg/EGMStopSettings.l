;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::EGMStopSettings)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'EGMStopSettings (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::EGMSTOPSETTINGS")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::EGMSTOPSETTINGS"))

(in-package "ROS")
;;//! \htmlinclude EGMStopSettings.msg.html


(defclass abb_rapid_sm_addin_msgs::EGMStopSettings
  :super ros::object
  :slots (_ramp_out_time ))

(defmethod abb_rapid_sm_addin_msgs::EGMStopSettings
  (:init
   (&key
    ((:ramp_out_time __ramp_out_time) 0.0)
    )
   (send-super :init)
   (setq _ramp_out_time (float __ramp_out_time))
   self)
  (:ramp_out_time
   (&optional __ramp_out_time)
   (if __ramp_out_time (setq _ramp_out_time __ramp_out_time)) _ramp_out_time)
  (:serialization-length
   ()
   (+
    ;; float32 _ramp_out_time
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _ramp_out_time
       (sys::poke _ramp_out_time (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _ramp_out_time
     (setq _ramp_out_time (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::EGMStopSettings :md5sum-) "0e3716aa777ece435e2f7343649fe19b")
(setf (get abb_rapid_sm_addin_msgs::EGMStopSettings :datatype-) "abb_rapid_sm_addin_msgs/EGMStopSettings")
(setf (get abb_rapid_sm_addin_msgs::EGMStopSettings :definition-)
      "#-------------------------------------------------------------------------------
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



(provide :abb_rapid_sm_addin_msgs/EGMStopSettings "0e3716aa777ece435e2f7343649fe19b")


