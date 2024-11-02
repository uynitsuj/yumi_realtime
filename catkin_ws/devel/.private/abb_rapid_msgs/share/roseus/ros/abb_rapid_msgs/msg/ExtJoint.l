;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::ExtJoint)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'ExtJoint (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::EXTJOINT")
  (make-package "ABB_RAPID_MSGS::EXTJOINT"))

(in-package "ROS")
;;//! \htmlinclude ExtJoint.msg.html


(defclass abb_rapid_msgs::ExtJoint
  :super ros::object
  :slots (_eax_a _eax_b _eax_c _eax_d _eax_e _eax_f ))

(defmethod abb_rapid_msgs::ExtJoint
  (:init
   (&key
    ((:eax_a __eax_a) 0.0)
    ((:eax_b __eax_b) 0.0)
    ((:eax_c __eax_c) 0.0)
    ((:eax_d __eax_d) 0.0)
    ((:eax_e __eax_e) 0.0)
    ((:eax_f __eax_f) 0.0)
    )
   (send-super :init)
   (setq _eax_a (float __eax_a))
   (setq _eax_b (float __eax_b))
   (setq _eax_c (float __eax_c))
   (setq _eax_d (float __eax_d))
   (setq _eax_e (float __eax_e))
   (setq _eax_f (float __eax_f))
   self)
  (:eax_a
   (&optional __eax_a)
   (if __eax_a (setq _eax_a __eax_a)) _eax_a)
  (:eax_b
   (&optional __eax_b)
   (if __eax_b (setq _eax_b __eax_b)) _eax_b)
  (:eax_c
   (&optional __eax_c)
   (if __eax_c (setq _eax_c __eax_c)) _eax_c)
  (:eax_d
   (&optional __eax_d)
   (if __eax_d (setq _eax_d __eax_d)) _eax_d)
  (:eax_e
   (&optional __eax_e)
   (if __eax_e (setq _eax_e __eax_e)) _eax_e)
  (:eax_f
   (&optional __eax_f)
   (if __eax_f (setq _eax_f __eax_f)) _eax_f)
  (:serialization-length
   ()
   (+
    ;; float32 _eax_a
    4
    ;; float32 _eax_b
    4
    ;; float32 _eax_c
    4
    ;; float32 _eax_d
    4
    ;; float32 _eax_e
    4
    ;; float32 _eax_f
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _eax_a
       (sys::poke _eax_a (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _eax_b
       (sys::poke _eax_b (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _eax_c
       (sys::poke _eax_c (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _eax_d
       (sys::poke _eax_d (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _eax_e
       (sys::poke _eax_e (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _eax_f
       (sys::poke _eax_f (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _eax_a
     (setq _eax_a (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _eax_b
     (setq _eax_b (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _eax_c
     (setq _eax_c (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _eax_d
     (setq _eax_d (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _eax_e
     (setq _eax_e (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _eax_f
     (setq _eax_f (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_msgs::ExtJoint :md5sum-) "f8550d9e82bcfae84c73de38fde07b03")
(setf (get abb_rapid_msgs::ExtJoint :datatype-) "abb_rapid_msgs/ExtJoint")
(setf (get abb_rapid_msgs::ExtJoint :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"extjoint\" (position of external joints).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The position of the external logical axis \"a\"-\"f\",
# expressed in degrees or mm (depending on the type of axis).
float32 eax_a
float32 eax_b
float32 eax_c
float32 eax_d
float32 eax_e
float32 eax_f

")



(provide :abb_rapid_msgs/ExtJoint "f8550d9e82bcfae84c73de38fde07b03")


