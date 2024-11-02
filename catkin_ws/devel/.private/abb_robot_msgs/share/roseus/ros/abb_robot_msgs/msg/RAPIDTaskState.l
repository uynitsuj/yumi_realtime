;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::RAPIDTaskState)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'RAPIDTaskState (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE")
  (make-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))

(in-package "ROS")
;;//! \htmlinclude RAPIDTaskState.msg.html


(intern "*EXECUTION_STATE_UNKNOWN*" (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(shadow '*EXECUTION_STATE_UNKNOWN* (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(defconstant abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_UNKNOWN* 1)
(intern "*EXECUTION_STATE_READY*" (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(shadow '*EXECUTION_STATE_READY* (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(defconstant abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_READY* 2)
(intern "*EXECUTION_STATE_STOPPED*" (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(shadow '*EXECUTION_STATE_STOPPED* (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(defconstant abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_STOPPED* 3)
(intern "*EXECUTION_STATE_STARTED*" (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(shadow '*EXECUTION_STATE_STARTED* (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(defconstant abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_STARTED* 4)
(intern "*EXECUTION_STATE_UNINITIALIZED*" (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(shadow '*EXECUTION_STATE_UNINITIALIZED* (find-package "ABB_ROBOT_MSGS::RAPIDTASKSTATE"))
(defconstant abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_UNINITIALIZED* 5)

(defun abb_robot_msgs::RAPIDTaskState-to-symbol (const)
  (cond
        ((= const 1) 'abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_UNKNOWN*)
        ((= const 2) 'abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_READY*)
        ((= const 3) 'abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_STOPPED*)
        ((= const 4) 'abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_STARTED*)
        ((= const 5) 'abb_robot_msgs::RAPIDTaskState::*EXECUTION_STATE_UNINITIALIZED*)
        (t nil)))

(defclass abb_robot_msgs::RAPIDTaskState
  :super ros::object
  :slots (_name _activated _execution_state _motion_task ))

(defmethod abb_robot_msgs::RAPIDTaskState
  (:init
   (&key
    ((:name __name) "")
    ((:activated __activated) nil)
    ((:execution_state __execution_state) 0)
    ((:motion_task __motion_task) nil)
    )
   (send-super :init)
   (setq _name (string __name))
   (setq _activated __activated)
   (setq _execution_state (round __execution_state))
   (setq _motion_task __motion_task)
   self)
  (:name
   (&optional __name)
   (if __name (setq _name __name)) _name)
  (:activated
   (&optional (__activated :null))
   (if (not (eq __activated :null)) (setq _activated __activated)) _activated)
  (:execution_state
   (&optional __execution_state)
   (if __execution_state (setq _execution_state __execution_state)) _execution_state)
  (:motion_task
   (&optional (__motion_task :null))
   (if (not (eq __motion_task :null)) (setq _motion_task __motion_task)) _motion_task)
  (:serialization-length
   ()
   (+
    ;; string _name
    4 (length _name)
    ;; bool _activated
    1
    ;; uint8 _execution_state
    1
    ;; bool _motion_task
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
     ;; uint8 _execution_state
       (write-byte _execution_state s)
     ;; bool _motion_task
       (if _motion_task (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _activated
     (setq _activated (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; uint8 _execution_state
     (setq _execution_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; bool _motion_task
     (setq _motion_task (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get abb_robot_msgs::RAPIDTaskState :md5sum-) "37a1a82a542dcaea43aef8b824e26614")
(setf (get abb_robot_msgs::RAPIDTaskState :datatype-) "abb_robot_msgs/RAPIDTaskState")
(setf (get abb_robot_msgs::RAPIDTaskState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of a RAPID task defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# RAPID execution states:
uint8 EXECUTION_STATE_UNKNOWN       = 1
uint8 EXECUTION_STATE_READY         = 2
uint8 EXECUTION_STATE_STOPPED       = 3
uint8 EXECUTION_STATE_STARTED       = 4
uint8 EXECUTION_STATE_UNINITIALIZED = 5

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The RAPID task's name.
string name

# The RAPID task's activated/deactivated status.
bool activated

# The RAPID task's execution state.
uint8 execution_state

# The RAPID task is a motion task.
bool motion_task

")



(provide :abb_robot_msgs/RAPIDTaskState "37a1a82a542dcaea43aef8b824e26614")


