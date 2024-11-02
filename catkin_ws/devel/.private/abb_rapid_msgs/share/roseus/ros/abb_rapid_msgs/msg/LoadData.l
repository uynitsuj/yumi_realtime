;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::LoadData)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'LoadData (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::LOADDATA")
  (make-package "ABB_RAPID_MSGS::LOADDATA"))

(in-package "ROS")
;;//! \htmlinclude LoadData.msg.html


(defclass abb_rapid_msgs::LoadData
  :super ros::object
  :slots (_mass _cog _aom _ix _iy _iz ))

(defmethod abb_rapid_msgs::LoadData
  (:init
   (&key
    ((:mass __mass) 0.0)
    ((:cog __cog) (instance abb_rapid_msgs::Pos :init))
    ((:aom __aom) (instance abb_rapid_msgs::Orient :init))
    ((:ix __ix) 0.0)
    ((:iy __iy) 0.0)
    ((:iz __iz) 0.0)
    )
   (send-super :init)
   (setq _mass (float __mass))
   (setq _cog __cog)
   (setq _aom __aom)
   (setq _ix (float __ix))
   (setq _iy (float __iy))
   (setq _iz (float __iz))
   self)
  (:mass
   (&optional __mass)
   (if __mass (setq _mass __mass)) _mass)
  (:cog
   (&rest __cog)
   (if (keywordp (car __cog))
       (send* _cog __cog)
     (progn
       (if __cog (setq _cog (car __cog)))
       _cog)))
  (:aom
   (&rest __aom)
   (if (keywordp (car __aom))
       (send* _aom __aom)
     (progn
       (if __aom (setq _aom (car __aom)))
       _aom)))
  (:ix
   (&optional __ix)
   (if __ix (setq _ix __ix)) _ix)
  (:iy
   (&optional __iy)
   (if __iy (setq _iy __iy)) _iy)
  (:iz
   (&optional __iz)
   (if __iz (setq _iz __iz)) _iz)
  (:serialization-length
   ()
   (+
    ;; float32 _mass
    4
    ;; abb_rapid_msgs/Pos _cog
    (send _cog :serialization-length)
    ;; abb_rapid_msgs/Orient _aom
    (send _aom :serialization-length)
    ;; float32 _ix
    4
    ;; float32 _iy
    4
    ;; float32 _iz
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _mass
       (sys::poke _mass (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; abb_rapid_msgs/Pos _cog
       (send _cog :serialize s)
     ;; abb_rapid_msgs/Orient _aom
       (send _aom :serialize s)
     ;; float32 _ix
       (sys::poke _ix (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _iy
       (sys::poke _iy (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _iz
       (sys::poke _iz (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _mass
     (setq _mass (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; abb_rapid_msgs/Pos _cog
     (send _cog :deserialize buf ptr-) (incf ptr- (send _cog :serialization-length))
   ;; abb_rapid_msgs/Orient _aom
     (send _aom :deserialize buf ptr-) (incf ptr- (send _aom :serialization-length))
   ;; float32 _ix
     (setq _ix (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _iy
     (setq _iy (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _iz
     (setq _iz (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get abb_rapid_msgs::LoadData :md5sum-) "dfe575484752eff5e837f54d2fd939e4")
(setf (get abb_rapid_msgs::LoadData :datatype-) "abb_rapid_msgs/LoadData")
(setf (get abb_rapid_msgs::LoadData :definition-)
      "#-------------------------------------------------------------------------------
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



(provide :abb_rapid_msgs/LoadData "dfe575484752eff5e837f54d2fd939e4")


