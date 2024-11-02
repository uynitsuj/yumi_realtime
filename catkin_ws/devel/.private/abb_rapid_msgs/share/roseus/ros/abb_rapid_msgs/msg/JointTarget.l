;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::JointTarget)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'JointTarget (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::JOINTTARGET")
  (make-package "ABB_RAPID_MSGS::JOINTTARGET"))

(in-package "ROS")
;;//! \htmlinclude JointTarget.msg.html


(defclass abb_rapid_msgs::JointTarget
  :super ros::object
  :slots (_robax _extax ))

(defmethod abb_rapid_msgs::JointTarget
  (:init
   (&key
    ((:robax __robax) (instance abb_rapid_msgs::RobJoint :init))
    ((:extax __extax) (instance abb_rapid_msgs::ExtJoint :init))
    )
   (send-super :init)
   (setq _robax __robax)
   (setq _extax __extax)
   self)
  (:robax
   (&rest __robax)
   (if (keywordp (car __robax))
       (send* _robax __robax)
     (progn
       (if __robax (setq _robax (car __robax)))
       _robax)))
  (:extax
   (&rest __extax)
   (if (keywordp (car __extax))
       (send* _extax __extax)
     (progn
       (if __extax (setq _extax (car __extax)))
       _extax)))
  (:serialization-length
   ()
   (+
    ;; abb_rapid_msgs/RobJoint _robax
    (send _robax :serialization-length)
    ;; abb_rapid_msgs/ExtJoint _extax
    (send _extax :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_rapid_msgs/RobJoint _robax
       (send _robax :serialize s)
     ;; abb_rapid_msgs/ExtJoint _extax
       (send _extax :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_rapid_msgs/RobJoint _robax
     (send _robax :deserialize buf ptr-) (incf ptr- (send _robax :serialization-length))
   ;; abb_rapid_msgs/ExtJoint _extax
     (send _extax :deserialize buf ptr-) (incf ptr- (send _extax :serialization-length))
   ;;
   self)
  )

(setf (get abb_rapid_msgs::JointTarget :md5sum-) "a80f8974c2ebed9e2ef462fcc2a28a38")
(setf (get abb_rapid_msgs::JointTarget :datatype-) "abb_rapid_msgs/JointTarget")
(setf (get abb_rapid_msgs::JointTarget :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"jointtarget\" (joint position data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Axis positions of the robot axes in degrees.
RobJoint robax

# The position of the external axes.
ExtJoint extax

================================================================================
MSG: abb_rapid_msgs/RobJoint
#-------------------------------------------------------------------------------
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

================================================================================
MSG: abb_rapid_msgs/ExtJoint
#-------------------------------------------------------------------------------
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



(provide :abb_rapid_msgs/JointTarget "a80f8974c2ebed9e2ef462fcc2a28a38")


