;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::Orient)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'Orient (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::ORIENT")
  (make-package "ABB_RAPID_MSGS::ORIENT"))

(in-package "ROS")
;;//! \htmlinclude Orient.msg.html


(defclass abb_rapid_msgs::Orient
  :super ros::object
  :slots (_q1 _q2 _q3 _q4 ))

(defmethod abb_rapid_msgs::Orient
  (:init
   (&key
    ((:q1 __q1) 0.0)
    ((:q2 __q2) 0.0)
    ((:q3 __q3) 0.0)
    ((:q4 __q4) 0.0)
    )
   (send-super :init)
   (setq _q1 (float __q1))
   (setq _q2 (float __q2))
   (setq _q3 (float __q3))
   (setq _q4 (float __q4))
   self)
  (:q1
   (&optional __q1)
   (if __q1 (setq _q1 __q1)) _q1)
  (:q2
   (&optional __q2)
   (if __q2 (setq _q2 __q2)) _q2)
  (:q3
   (&optional __q3)
   (if __q3 (setq _q3 __q3)) _q3)
  (:q4
   (&optional __q4)
   (if __q4 (setq _q4 __q4)) _q4)
  (:serialization-length
   ()
   (+
    ;; float32 _q1
    4
    ;; float32 _q2
    4
    ;; float32 _q3
    4
    ;; float32 _q4
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _q1
       (sys::poke _q1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q2
       (sys::poke _q2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q3
       (sys::poke _q3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _q4
       (sys::poke _q4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _q1
     (setq _q1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q2
     (setq _q2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q3
     (setq _q3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _q4
     (setq _q4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_msgs::Orient :md5sum-) "69cec82442f682ecfe3c0ec1306ced82")
(setf (get abb_rapid_msgs::Orient :datatype-) "abb_rapid_msgs/Orient")
(setf (get abb_rapid_msgs::Orient :definition-)
      "#-------------------------------------------------------------------------------
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



(provide :abb_rapid_msgs/Orient "69cec82442f682ecfe3c0ec1306ced82")


