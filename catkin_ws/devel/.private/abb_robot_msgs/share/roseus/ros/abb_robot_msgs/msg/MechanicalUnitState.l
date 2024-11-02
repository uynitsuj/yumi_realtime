;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::MechanicalUnitState)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'MechanicalUnitState (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::MECHANICALUNITSTATE")
  (make-package "ABB_ROBOT_MSGS::MECHANICALUNITSTATE"))

(in-package "ROS")
;;//! \htmlinclude MechanicalUnitState.msg.html


(defclass abb_robot_msgs::MechanicalUnitState
  :super ros::object
  :slots (_name _activated ))

(defmethod abb_robot_msgs::MechanicalUnitState
  (:init
   (&key
    ((:name __name) "")
    ((:activated __activated) nil)
    )
   (send-super :init)
   (setq _name (string __name))
   (setq _activated __activated)
   self)
  (:name
   (&optional __name)
   (if __name (setq _name __name)) _name)
  (:activated
   (&optional (__activated :null))
   (if (not (eq __activated :null)) (setq _activated __activated)) _activated)
  (:serialization-length
   ()
   (+
    ;; string _name
    4 (length _name)
    ;; bool _activated
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _name
       (write-long (length _name) s) (princ _name s)
     ;; bool _activated
       (if _activated (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _activated
     (setq _activated (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get abb_robot_msgs::MechanicalUnitState :md5sum-) "add093e2c7fdce39ce8a0c1e573668e8")
(setf (get abb_robot_msgs::MechanicalUnitState :datatype-) "abb_robot_msgs/MechanicalUnitState")
(setf (get abb_robot_msgs::MechanicalUnitState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of a mechanical unit defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The mechanical unit's name.
string name

# The mechanical unit's activated/deactivated status.
bool activated

")



(provide :abb_robot_msgs/MechanicalUnitState "add093e2c7fdce39ce8a0c1e573668e8")


