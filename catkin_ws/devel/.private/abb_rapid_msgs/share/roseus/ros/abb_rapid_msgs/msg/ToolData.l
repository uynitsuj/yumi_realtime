;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::ToolData)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'ToolData (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::TOOLDATA")
  (make-package "ABB_RAPID_MSGS::TOOLDATA"))

(in-package "ROS")
;;//! \htmlinclude ToolData.msg.html


(defclass abb_rapid_msgs::ToolData
  :super ros::object
  :slots (_robhold _tframe _tload ))

(defmethod abb_rapid_msgs::ToolData
  (:init
   (&key
    ((:robhold __robhold) nil)
    ((:tframe __tframe) (instance abb_rapid_msgs::Pose :init))
    ((:tload __tload) (instance abb_rapid_msgs::LoadData :init))
    )
   (send-super :init)
   (setq _robhold __robhold)
   (setq _tframe __tframe)
   (setq _tload __tload)
   self)
  (:robhold
   (&optional (__robhold :null))
   (if (not (eq __robhold :null)) (setq _robhold __robhold)) _robhold)
  (:tframe
   (&rest __tframe)
   (if (keywordp (car __tframe))
       (send* _tframe __tframe)
     (progn
       (if __tframe (setq _tframe (car __tframe)))
       _tframe)))
  (:tload
   (&rest __tload)
   (if (keywordp (car __tload))
       (send* _tload __tload)
     (progn
       (if __tload (setq _tload (car __tload)))
       _tload)))
  (:serialization-length
   ()
   (+
    ;; bool _robhold
    1
    ;; abb_rapid_msgs/Pose _tframe
    (send _tframe :serialization-length)
    ;; abb_rapid_msgs/LoadData _tload
    (send _tload :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _robhold
       (if _robhold (write-byte -1 s) (write-byte 0 s))
     ;; abb_rapid_msgs/Pose _tframe
       (send _tframe :serialize s)
     ;; abb_rapid_msgs/LoadData _tload
       (send _tload :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _robhold
     (setq _robhold (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; abb_rapid_msgs/Pose _tframe
     (send _tframe :deserialize buf ptr-) (incf ptr- (send _tframe :serialization-length))
   ;; abb_rapid_msgs/LoadData _tload
     (send _tload :deserialize buf ptr-) (incf ptr- (send _tload :serialization-length))
   ;;
   self)
  )

(setf (get abb_rapid_msgs::ToolData :md5sum-) "2d526e4dd17e28147b27f746cf88c5b6")
(setf (get abb_rapid_msgs::ToolData :datatype-) "abb_rapid_msgs/ToolData")
(setf (get abb_rapid_msgs::ToolData :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"tooldata\" (tool data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Defines whether or not the robot is holding the tool:
# - TRUE: The robot is holding the tool.
# - FALSE: The robot is not holding the tool, that is, a stationary tool.
bool robhold

# The tool coordinate system, that is:
# - The position of the TCP (x, y and z) in mm,
#   expressed in the wrist coordinate system (tool0).
# - The orientation of the tool coordinate system,
#   expressed in the wrist coordinate system.
Pose tframe

# The load of the tool (if robot held tool).
# The load of the gripper holding the work object (if stationary tool).
LoadData tload

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

================================================================================
MSG: abb_rapid_msgs/LoadData
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"loaddata\" (load data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The mass (weight) of the load in kg.
float32 mass

# The center of gravity of the payload, expressed in mm,
# in the tool coordinate system if the robot is holding the tool.
#
# If a stationary tool is used then the center of gravity for the payload
# held by the gripper is expressed in the object frame of the work object
# coordinate system moved by the robot.
Pos cog

# The orientation of the axes of moment.
#
# These are the principal axes of the payload
# moment of inertia with origin in cog.
#
# If the robot is holding the tool, the axes of moment
# are expressed in the tool coordinate system.
Orient aom

# The moment of inertia of the load around the x-, y- and z-axis,
# expressed in kgm^2.
float32 ix
float32 iy
float32 iz

")



(provide :abb_rapid_msgs/ToolData "2d526e4dd17e28147b27f746cf88c5b6")


