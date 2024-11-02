;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::EGMSetupSettings)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'EGMSetupSettings (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::EGMSETUPSETTINGS")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::EGMSETUPSETTINGS"))

(in-package "ROS")
;;//! \htmlinclude EGMSetupSettings.msg.html


(defclass abb_rapid_sm_addin_msgs::EGMSetupSettings
  :super ros::object
  :slots (_use_filtering _comm_timeout ))

(defmethod abb_rapid_sm_addin_msgs::EGMSetupSettings
  (:init
   (&key
    ((:use_filtering __use_filtering) nil)
    ((:comm_timeout __comm_timeout) 0.0)
    )
   (send-super :init)
   (setq _use_filtering __use_filtering)
   (setq _comm_timeout (float __comm_timeout))
   self)
  (:use_filtering
   (&optional (__use_filtering :null))
   (if (not (eq __use_filtering :null)) (setq _use_filtering __use_filtering)) _use_filtering)
  (:comm_timeout
   (&optional __comm_timeout)
   (if __comm_timeout (setq _comm_timeout __comm_timeout)) _comm_timeout)
  (:serialization-length
   ()
   (+
    ;; bool _use_filtering
    1
    ;; float32 _comm_timeout
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _use_filtering
       (if _use_filtering (write-byte -1 s) (write-byte 0 s))
     ;; float32 _comm_timeout
       (sys::poke _comm_timeout (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _use_filtering
     (setq _use_filtering (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _comm_timeout
     (setq _comm_timeout (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::EGMSetupSettings :md5sum-) "13bb9a91c166ef61bfdd5d16c6d9ac65")
(setf (get abb_rapid_sm_addin_msgs::EGMSetupSettings :datatype-) "abb_rapid_sm_addin_msgs/EGMSetupSettings")
(setf (get abb_rapid_sm_addin_msgs::EGMSetupSettings :definition-)
      "#-------------------------------------------------------------------------------
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

")



(provide :abb_rapid_sm_addin_msgs/EGMSetupSettings "13bb9a91c166ef61bfdd5d16c6d9ac65")


