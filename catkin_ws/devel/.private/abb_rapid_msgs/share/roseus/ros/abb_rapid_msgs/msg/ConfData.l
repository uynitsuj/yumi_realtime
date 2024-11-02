;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::ConfData)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'ConfData (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::CONFDATA")
  (make-package "ABB_RAPID_MSGS::CONFDATA"))

(in-package "ROS")
;;//! \htmlinclude ConfData.msg.html


(defclass abb_rapid_msgs::ConfData
  :super ros::object
  :slots (_cf1 _cf4 _cf6 _cfx ))

(defmethod abb_rapid_msgs::ConfData
  (:init
   (&key
    ((:cf1 __cf1) 0.0)
    ((:cf4 __cf4) 0.0)
    ((:cf6 __cf6) 0.0)
    ((:cfx __cfx) 0.0)
    )
   (send-super :init)
   (setq _cf1 (float __cf1))
   (setq _cf4 (float __cf4))
   (setq _cf6 (float __cf6))
   (setq _cfx (float __cfx))
   self)
  (:cf1
   (&optional __cf1)
   (if __cf1 (setq _cf1 __cf1)) _cf1)
  (:cf4
   (&optional __cf4)
   (if __cf4 (setq _cf4 __cf4)) _cf4)
  (:cf6
   (&optional __cf6)
   (if __cf6 (setq _cf6 __cf6)) _cf6)
  (:cfx
   (&optional __cfx)
   (if __cfx (setq _cfx __cfx)) _cfx)
  (:serialization-length
   ()
   (+
    ;; float32 _cf1
    4
    ;; float32 _cf4
    4
    ;; float32 _cf6
    4
    ;; float32 _cfx
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _cf1
       (sys::poke _cf1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _cf4
       (sys::poke _cf4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _cf6
       (sys::poke _cf6 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _cfx
       (sys::poke _cfx (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _cf1
     (setq _cf1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _cf4
     (setq _cf4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _cf6
     (setq _cf6 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _cfx
     (setq _cfx (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_msgs::ConfData :md5sum-) "21f1129f94cd63dd0cdf33be2bddc75e")
(setf (get abb_rapid_msgs::ConfData :datatype-) "abb_rapid_msgs/ConfData")
(setf (get abb_rapid_msgs::ConfData :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"confdata\" (robot configuration data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Rotating axis:
# - The current quadrant of axis 1, 4 and 6,
#   expressed as a positive or negative integer.
# Linear axis:
# - The current meter interval of axis 1, 4 and 6,
#   expressed as a positive or negative integer.
float32 cf1
float32 cf4
float32 cf6

# Rotating axis:
# - For serial link robots, the current robot configuration,
#   expressed as an integer in the range from 0 to 7.
# - For SCARA robots, the current robot configuration,
#   expressed as an integer in the range from 0 to 1.
# - For 7-axis robots, the current robot configuration,
#   expressed as an integer in the range from 0 to 7.
# - For paint robots, the current quadrant of axis 5,
#   expressed as a positive or negative integer.
# - For other robots, using the current quadrant of axis 2,
#   expressed as a positive or negative integer.
# Linear axis:
# - The current meter interval of axis 2,
#   expressed as a positive or negative integer.
float32 cfx

")



(provide :abb_rapid_msgs/ConfData "21f1129f94cd63dd0cdf33be2bddc75e")


