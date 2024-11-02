;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::RobJoint)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'RobJoint (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::ROBJOINT")
  (make-package "ABB_RAPID_MSGS::ROBJOINT"))

(in-package "ROS")
;;//! \htmlinclude RobJoint.msg.html


(defclass abb_rapid_msgs::RobJoint
  :super ros::object
  :slots (_rax_1 _rax_2 _rax_3 _rax_4 _rax_5 _rax_6 ))

(defmethod abb_rapid_msgs::RobJoint
  (:init
   (&key
    ((:rax_1 __rax_1) 0.0)
    ((:rax_2 __rax_2) 0.0)
    ((:rax_3 __rax_3) 0.0)
    ((:rax_4 __rax_4) 0.0)
    ((:rax_5 __rax_5) 0.0)
    ((:rax_6 __rax_6) 0.0)
    )
   (send-super :init)
   (setq _rax_1 (float __rax_1))
   (setq _rax_2 (float __rax_2))
   (setq _rax_3 (float __rax_3))
   (setq _rax_4 (float __rax_4))
   (setq _rax_5 (float __rax_5))
   (setq _rax_6 (float __rax_6))
   self)
  (:rax_1
   (&optional __rax_1)
   (if __rax_1 (setq _rax_1 __rax_1)) _rax_1)
  (:rax_2
   (&optional __rax_2)
   (if __rax_2 (setq _rax_2 __rax_2)) _rax_2)
  (:rax_3
   (&optional __rax_3)
   (if __rax_3 (setq _rax_3 __rax_3)) _rax_3)
  (:rax_4
   (&optional __rax_4)
   (if __rax_4 (setq _rax_4 __rax_4)) _rax_4)
  (:rax_5
   (&optional __rax_5)
   (if __rax_5 (setq _rax_5 __rax_5)) _rax_5)
  (:rax_6
   (&optional __rax_6)
   (if __rax_6 (setq _rax_6 __rax_6)) _rax_6)
  (:serialization-length
   ()
   (+
    ;; float32 _rax_1
    4
    ;; float32 _rax_2
    4
    ;; float32 _rax_3
    4
    ;; float32 _rax_4
    4
    ;; float32 _rax_5
    4
    ;; float32 _rax_6
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _rax_1
       (sys::poke _rax_1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rax_2
       (sys::poke _rax_2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rax_3
       (sys::poke _rax_3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rax_4
       (sys::poke _rax_4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rax_5
       (sys::poke _rax_5 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rax_6
       (sys::poke _rax_6 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _rax_1
     (setq _rax_1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rax_2
     (setq _rax_2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rax_3
     (setq _rax_3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rax_4
     (setq _rax_4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rax_5
     (setq _rax_5 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rax_6
     (setq _rax_6 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_msgs::RobJoint :md5sum-) "24a88117c4b385974e50d44f6663eb2a")
(setf (get abb_rapid_msgs::RobJoint :datatype-) "abb_rapid_msgs/RobJoint")
(setf (get abb_rapid_msgs::RobJoint :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"robjoint\" (joint position of robot axes).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The position of robot axis 1-6 in degrees from the calibration position.
float32 rax_1
float32 rax_2
float32 rax_3
float32 rax_4
float32 rax_5
float32 rax_6

")



(provide :abb_rapid_msgs/RobJoint "24a88117c4b385974e50d44f6663eb2a")


