;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_msgs::WObjData)
  (if (not (find-package "ABB_RAPID_MSGS"))
    (make-package "ABB_RAPID_MSGS"))
  (shadow 'WObjData (find-package "ABB_RAPID_MSGS")))
(unless (find-package "ABB_RAPID_MSGS::WOBJDATA")
  (make-package "ABB_RAPID_MSGS::WOBJDATA"))

(in-package "ROS")
;;//! \htmlinclude WObjData.msg.html


(defclass abb_rapid_msgs::WObjData
  :super ros::object
  :slots (_robhold _ufprog _ufmec _uframe _oframe ))

(defmethod abb_rapid_msgs::WObjData
  (:init
   (&key
    ((:robhold __robhold) nil)
    ((:ufprog __ufprog) nil)
    ((:ufmec __ufmec) "")
    ((:uframe __uframe) (instance abb_rapid_msgs::Pose :init))
    ((:oframe __oframe) (instance abb_rapid_msgs::Pose :init))
    )
   (send-super :init)
   (setq _robhold __robhold)
   (setq _ufprog __ufprog)
   (setq _ufmec (string __ufmec))
   (setq _uframe __uframe)
   (setq _oframe __oframe)
   self)
  (:robhold
   (&optional (__robhold :null))
   (if (not (eq __robhold :null)) (setq _robhold __robhold)) _robhold)
  (:ufprog
   (&optional (__ufprog :null))
   (if (not (eq __ufprog :null)) (setq _ufprog __ufprog)) _ufprog)
  (:ufmec
   (&optional __ufmec)
   (if __ufmec (setq _ufmec __ufmec)) _ufmec)
  (:uframe
   (&rest __uframe)
   (if (keywordp (car __uframe))
       (send* _uframe __uframe)
     (progn
       (if __uframe (setq _uframe (car __uframe)))
       _uframe)))
  (:oframe
   (&rest __oframe)
   (if (keywordp (car __oframe))
       (send* _oframe __oframe)
     (progn
       (if __oframe (setq _oframe (car __oframe)))
       _oframe)))
  (:serialization-length
   ()
   (+
    ;; bool _robhold
    1
    ;; bool _ufprog
    1
    ;; string _ufmec
    4 (length _ufmec)
    ;; abb_rapid_msgs/Pose _uframe
    (send _uframe :serialization-length)
    ;; abb_rapid_msgs/Pose _oframe
    (send _oframe :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _robhold
       (if _robhold (write-byte -1 s) (write-byte 0 s))
     ;; bool _ufprog
       (if _ufprog (write-byte -1 s) (write-byte 0 s))
     ;; string _ufmec
       (write-long (length _ufmec) s) (princ _ufmec s)
     ;; abb_rapid_msgs/Pose _uframe
       (send _uframe :serialize s)
     ;; abb_rapid_msgs/Pose _oframe
       (send _oframe :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _robhold
     (setq _robhold (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _ufprog
     (setq _ufprog (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _ufmec
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _ufmec (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; abb_rapid_msgs/Pose _uframe
     (send _uframe :deserialize buf ptr-) (incf ptr- (send _uframe :serialization-length))
   ;; abb_rapid_msgs/Pose _oframe
     (send _oframe :deserialize buf ptr-) (incf ptr- (send _oframe :serialization-length))
   ;;
   self)
  )

(setf (get abb_rapid_msgs::WObjData :md5sum-) "eb6fd21e8b900ca32b7fe6c07697ac07")
(setf (get abb_rapid_msgs::WObjData :datatype-) "abb_rapid_msgs/WObjData")
(setf (get abb_rapid_msgs::WObjData :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the
#   RAPID data type \"wobjdata\" (work object data).
#
# Note:
#   Please see the \"Technical reference manual - RAPID Instructions,
#   Functions and Data types\" for more details.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Defines whether or not the robot in the actual
# program task is holding the work object:
# - TRUE: The robot is holding the work object,
#         i.e. using a stationary tool.
# - FALSE: The robot is not holding the work object,
#          i.e. the robot is holding the tool.
bool robhold

# Defines whether or not a fixed user coordinate system is used:
# - TRUE: Fixed user coordinate system.
# - FALSE: Movable user coordinate system,
#          i.e. coordinated external axes are used.
#          Also to be used in a MultiMove system in
#          semicoordinated or synchronized coordinated mode.
bool ufprog

# The mechanical unit with which the robot movements are coordinated.
#
# Only specified in the case of movable user
# coordinate systems (ufprog is FALSE).
string ufmec

# The user coordinate system, i.e. the position
# of the current work surface or fixture.
Pose uframe

# The object coordinate system, i.e. the position
# of the current work object.
Pose oframe

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

")



(provide :abb_rapid_msgs/WObjData "eb6fd21e8b900ca32b7fe6c07697ac07")


