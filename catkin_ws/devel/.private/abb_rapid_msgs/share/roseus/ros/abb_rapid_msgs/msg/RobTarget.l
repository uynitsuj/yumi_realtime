;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::RobTarget)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'RobTarget (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::ROBTARGET")
  (make-package "ABB_RAPID_MSGS::ROBTARGET"))

(in-package "ROS")
;;//! \htmlinclude RobTarget.msg.html


(defclass abb_rapid_msgs::RobTarget
  :super ros::object
  :slots (_trans _rot _robconf _extax ))

(defmethod abb_rapid_msgs::RobTarget
  (:init
   (&key
    ((:trans __trans) (instance abb_rapid_msgs::Pos :init))
    ((:rot __rot) (instance abb_rapid_msgs::Orient :init))
    ((:robconf __robconf) (instance abb_rapid_msgs::ConfData :init))
    ((:extax __extax) (instance abb_rapid_msgs::ExtJoint :init))
    )
   (send-super :init)
   (setq _trans __trans)
   (setq _rot __rot)
   (setq _robconf __robconf)
   (setq _extax __extax)
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
  (:robconf
   (&rest __robconf)
   (if (keywordp (car __robconf))
       (send* _robconf __robconf)
     (progn
       (if __robconf (setq _robconf (car __robconf)))
       _robconf)))
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
    ;; abb_rapid_msgs/Pos _trans
    (send _trans :serialization-length)
    ;; abb_rapid_msgs/Orient _rot
    (send _rot :serialization-length)
    ;; abb_rapid_msgs/ConfData _robconf
    (send _robconf :serialization-length)
    ;; abb_rapid_msgs/ExtJoint _extax
    (send _extax :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_rapid_msgs/Pos _trans
       (send _trans :serialize s)
     ;; abb_rapid_msgs/Orient _rot
       (send _rot :serialize s)
     ;; abb_rapid_msgs/ConfData _robconf
       (send _robconf :serialize s)
     ;; abb_rapid_msgs/ExtJoint _extax
       (send _extax :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_rapid_msgs/Pos _trans
     (send _trans :deserialize buf ptr-) (incf ptr- (send _trans :serialization-length))
   ;; abb_rapid_msgs/Orient _rot
     (send _rot :deserialize buf ptr-) (incf ptr- (send _rot :serialization-length))
   ;; abb_rapid_msgs/ConfData _robconf
     (send _robconf :deserialize buf ptr-) (incf ptr- (send _robconf :serialization-length))
   ;; abb_rapid_msgs/ExtJoint _extax
     (send _extax :deserialize buf ptr-) (incf ptr- (send _extax :serialization-length))
   ;;
   self)
  )

(setf (get abb_rapid_msgs::RobTarget :md5sum-) "31d5c046c0fae9eadd8133382f15db0a")
(setf (get abb_rapid_msgs::RobTarget :datatype-) "abb_rapid_msgs/RobTarget")
(setf (get abb_rapid_msgs::RobTarget :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"robtarget\" (position data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The position (x, y, and z) of the tool center point expressed in mm.
#
# The position is specified in relation to the current object coordinate system,
# including program displacement.
#
# If no work object is specified then this is the world coordinate system.
Pos trans

# The orientation of the tool, expressed in the
# form of a quaternion (q1, q2, q3, and q4).
#
# The orientation is specified in relation to the current
# object coordinate system including program displacement.
#
# If no work object is specified then this is the world coordinate system.
Orient rot

# The axis configuration of the robot (cf1, cf4, cf6, and cfx).
#
# This is defined in the form of the current quarter
# revolution of axis 1, axis 4, and axis 6.
#
# The first positive quarter revolution 0 to 90 degrees is defined as 0.
#
# The meaning of the component cfx is dependent on robot type.
ConfData robconf

# The position of the additional axes.
ExtJoint extax

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
MSG: abb_rapid_msgs/ConfData
#-------------------------------------------------------------------------------
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



(provide :abb_rapid_msgs/RobTarget "31d5c046c0fae9eadd8133382f15db0a")


