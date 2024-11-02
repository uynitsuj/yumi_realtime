;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::StateMachineState)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'StateMachineState (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))

(in-package "ROS")
;;//! \htmlinclude StateMachineState.msg.html


(intern "*SM_STATE_UNKNOWN*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*SM_STATE_UNKNOWN* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_UNKNOWN* 1)
(intern "*SM_STATE_IDLE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*SM_STATE_IDLE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_IDLE* 2)
(intern "*SM_STATE_INITIALIZE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*SM_STATE_INITIALIZE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_INITIALIZE* 3)
(intern "*SM_STATE_RUN_RAPID_ROUTINE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*SM_STATE_RUN_RAPID_ROUTINE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_RUN_RAPID_ROUTINE* 4)
(intern "*SM_STATE_RUN_EGM_ROUTINE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*SM_STATE_RUN_EGM_ROUTINE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_RUN_EGM_ROUTINE* 5)
(intern "*EGM_ACTION_UNKNOWN*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_UNKNOWN* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_UNKNOWN* 1)
(intern "*EGM_ACTION_NONE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_NONE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_NONE* 2)
(intern "*EGM_ACTION_RUN_JOINT*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_RUN_JOINT* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_RUN_JOINT* 3)
(intern "*EGM_ACTION_RUN_POSE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_RUN_POSE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_RUN_POSE* 4)
(intern "*EGM_ACTION_STOP*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_STOP* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_STOP* 5)
(intern "*EGM_ACTION_START_STREAM*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_START_STREAM* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_START_STREAM* 6)
(intern "*EGM_ACTION_STOP_STREAM*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(shadow '*EGM_ACTION_STOP_STREAM* (find-package "ABB_RAPID_SM_ADDIN_MSGS::STATEMACHINESTATE"))
(defconstant abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_STOP_STREAM* 7)

(defun abb_rapid_sm_addin_msgs::StateMachineState-to-symbol (const)
  (cond
        ((= const 1) 'abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_UNKNOWN*)
        ((= const 2) 'abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_IDLE*)
        ((= const 3) 'abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_INITIALIZE*)
        ((= const 4) 'abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_RUN_RAPID_ROUTINE*)
        ((= const 5) 'abb_rapid_sm_addin_msgs::StateMachineState::*SM_STATE_RUN_EGM_ROUTINE*)
        ((= const 1) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_UNKNOWN*)
        ((= const 2) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_NONE*)
        ((= const 3) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_RUN_JOINT*)
        ((= const 4) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_RUN_POSE*)
        ((= const 5) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_STOP*)
        ((= const 6) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_START_STREAM*)
        ((= const 7) 'abb_rapid_sm_addin_msgs::StateMachineState::*EGM_ACTION_STOP_STREAM*)
        (t nil)))

(defclass abb_rapid_sm_addin_msgs::StateMachineState
  :super ros::object
  :slots (_rapid_task _sm_state _egm_action ))

(defmethod abb_rapid_sm_addin_msgs::StateMachineState
  (:init
   (&key
    ((:rapid_task __rapid_task) "")
    ((:sm_state __sm_state) 0)
    ((:egm_action __egm_action) 0)
    )
   (send-super :init)
   (setq _rapid_task (string __rapid_task))
   (setq _sm_state (round __sm_state))
   (setq _egm_action (round __egm_action))
   self)
  (:rapid_task
   (&optional __rapid_task)
   (if __rapid_task (setq _rapid_task __rapid_task)) _rapid_task)
  (:sm_state
   (&optional __sm_state)
   (if __sm_state (setq _sm_state __sm_state)) _sm_state)
  (:egm_action
   (&optional __egm_action)
   (if __egm_action (setq _egm_action __egm_action)) _egm_action)
  (:serialization-length
   ()
   (+
    ;; string _rapid_task
    4 (length _rapid_task)
    ;; uint8 _sm_state
    1
    ;; uint8 _egm_action
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _rapid_task
       (write-long (length _rapid_task) s) (princ _rapid_task s)
     ;; uint8 _sm_state
       (write-byte _sm_state s)
     ;; uint8 _egm_action
       (write-byte _egm_action s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _rapid_task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _rapid_task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint8 _sm_state
     (setq _sm_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _egm_action
     (setq _egm_action (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::StateMachineState :md5sum-) "de6354e383149183de9c15ab3d871c7b")
(setf (get abb_rapid_sm_addin_msgs::StateMachineState :datatype-) "abb_rapid_sm_addin_msgs/StateMachineState")
(setf (get abb_rapid_sm_addin_msgs::StateMachineState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the state of an
#   instance of the RobotWare \"StateMachine Add-In\"'s RAPID implementation.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# Possible \"StateMachine Add-In\" RAPID states:
#
# Note: These enumeration values differ from the RAPID implementation
#       values to avoid conflict with default values in ROS messages.
uint8 SM_STATE_UNKNOWN           = 1
uint8 SM_STATE_IDLE              = 2
uint8 SM_STATE_INITIALIZE        = 3
uint8 SM_STATE_RUN_RAPID_ROUTINE = 4
uint8 SM_STATE_RUN_EGM_ROUTINE   = 5

# Possible \"StateMachine Add-In\" RAPID EGM actions:
#
# Note: These enumeration values differ from the RAPID implementation
#       values to avoid conflict with default values in ROS messages.
uint8 EGM_ACTION_UNKNOWN      = 1
uint8 EGM_ACTION_NONE         = 2
uint8 EGM_ACTION_RUN_JOINT    = 3
uint8 EGM_ACTION_RUN_POSE     = 4
uint8 EGM_ACTION_STOP         = 5
uint8 EGM_ACTION_START_STREAM = 6
uint8 EGM_ACTION_STOP_STREAM  = 7

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The RAPID task that runs the \"StateMachine Add-In\" instance.
string rapid_task

# The \"StateMachine Add-In\" instance's current state.
uint8 sm_state

# The \"StateMachine Add-In\" instance's current EGM action
# (only used if the RobotWare EGM option is present).
uint8 egm_action

")



(provide :abb_rapid_sm_addin_msgs/StateMachineState "de6354e383149183de9c15ab3d871c7b")


