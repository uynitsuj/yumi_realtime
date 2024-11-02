;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::Pose)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'Pose (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::POSE")
  (make-package "ABB_RAPID_MSGS::POSE"))

(in-package "ROS")
;;//! \htmlinclude Pose.msg.html


(defclass abb_rapid_msgs::Pose
  :super ros::object
  :slots (_trans _rot ))

(defmethod abb_rapid_msgs::Pose
  (:init
   (&key
    ((:trans __trans) (instance abb_rapid_msgs::Pos :init))
    ((:rot __rot) (instance abb_rapid_msgs::Orient :init))
    )
   (send-super :init)
   (setq _trans __trans)
   (setq _rot __rot)
   self)
  (:trans
   (&rest __trans)
   (if (keywordp (car __trans))
       (send* _trans __trans)
     (progn
       (if __trans (setq _trans (car __trans)))
       _trans)))
  (:rot
   (&rest __rot)
   (if (keywordp (car __rot))
       (send* _rot __rot)
     (progn
       (if __rot (setq _rot (car __rot)))
       _rot)))
  (:serialization-length
   ()
   (+
    ;; abb_rapid_msgs/Pos _trans
    (send _trans :serialization-length)
    ;; abb_rapid_msgs/Orient _rot
    (send _rot :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_rapid_msgs/Pos _trans
       (send _trans :serialize s)
     ;; abb_rapid_msgs/Orient _rot
       (send _rot :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_rapid_msgs/Pos _trans
     (send _trans :deserialize buf ptr-) (incf ptr- (send _trans :serialization-length))
   ;; abb_rapid_msgs/Orient _rot
     (send _rot :deserialize buf ptr-) (incf ptr- (send _rot :serialization-length))
   ;;
   self)
  )

(setf (get abb_rapid_msgs::Pose :md5sum-) "94c76c2f70e3c86d018090698fefbfdc")
(setf (get abb_rapid_msgs::Pose :datatype-) "abb_rapid_msgs/Pose")
(setf (get abb_rapid_msgs::Pose :definition-)
      "#-------------------------------------------------------------------------------
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



(provide :abb_rapid_msgs/Pose "94c76c2f70e3c86d018090698fefbfdc")


