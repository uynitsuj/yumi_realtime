;; Auto-generated. Do not edit!


(when (boundp 'abb_egm_msgs::EGMChannelState)
  (if (not (find-package "ABB_EGM_MSGS"))
    (make-package "ABB_EGM_MSGS"))
  (shadow 'EGMChannelState (find-package "ABB_EGM_MSGS")))
(unless (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE")
  (make-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))

(in-package "ROS")
;;//! \htmlinclude EGMChannelState.msg.html


(intern "*EGM_UNDEFINED*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*EGM_UNDEFINED* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*EGM_UNDEFINED* 1)
(intern "*EGM_ERROR*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*EGM_ERROR* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*EGM_ERROR* 2)
(intern "*EGM_STOPPED*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*EGM_STOPPED* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*EGM_STOPPED* 3)
(intern "*EGM_RUNNING*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*EGM_RUNNING* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*EGM_RUNNING* 4)
(intern "*MOTORS_UNDEFINED*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*MOTORS_UNDEFINED* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*MOTORS_UNDEFINED* 1)
(intern "*MOTORS_ON*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*MOTORS_ON* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*MOTORS_ON* 2)
(intern "*MOTORS_OFF*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*MOTORS_OFF* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*MOTORS_OFF* 3)
(intern "*RAPID_UNDEFINED*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*RAPID_UNDEFINED* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*RAPID_UNDEFINED* 1)
(intern "*RAPID_STOPPED*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*RAPID_STOPPED* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*RAPID_STOPPED* 2)
(intern "*RAPID_RUNNING*" (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(shadow '*RAPID_RUNNING* (find-package "ABB_EGM_MSGS::EGMCHANNELSTATE"))
(defconstant abb_egm_msgs::EGMChannelState::*RAPID_RUNNING* 3)

(defun abb_egm_msgs::EGMChannelState-to-symbol (const)
  (cond
        ((= const 1) 'abb_egm_msgs::EGMChannelState::*EGM_UNDEFINED*)
        ((= const 2) 'abb_egm_msgs::EGMChannelState::*EGM_ERROR*)
        ((= const 3) 'abb_egm_msgs::EGMChannelState::*EGM_STOPPED*)
        ((= const 4) 'abb_egm_msgs::EGMChannelState::*EGM_RUNNING*)
        ((= const 1) 'abb_egm_msgs::EGMChannelState::*MOTORS_UNDEFINED*)
        ((= const 2) 'abb_egm_msgs::EGMChannelState::*MOTORS_ON*)
        ((= const 3) 'abb_egm_msgs::EGMChannelState::*MOTORS_OFF*)
        ((= const 1) 'abb_egm_msgs::EGMChannelState::*RAPID_UNDEFINED*)
        ((= const 2) 'abb_egm_msgs::EGMChannelState::*RAPID_STOPPED*)
        ((= const 3) 'abb_egm_msgs::EGMChannelState::*RAPID_RUNNING*)
        (t nil)))

(defclass abb_egm_msgs::EGMChannelState
  :super ros::object
  :slots (_name _active _egm_convergence_met _egm_client_state _motor_state _rapid_execution_state _utilization_rate ))

(defmethod abb_egm_msgs::EGMChannelState
  (:init
   (&key
    ((:name __name) "")
    ((:active __active) nil)
    ((:egm_convergence_met __egm_convergence_met) nil)
    ((:egm_client_state __egm_client_state) 0)
    ((:motor_state __motor_state) 0)
    ((:rapid_execution_state __rapid_execution_state) 0)
    ((:utilization_rate __utilization_rate) 0.0)
    )
   (send-super :init)
   (setq _name (string __name))
   (setq _active __active)
   (setq _egm_convergence_met __egm_convergence_met)
   (setq _egm_client_state (round __egm_client_state))
   (setq _motor_state (round __motor_state))
   (setq _rapid_execution_state (round __rapid_execution_state))
   (setq _utilization_rate (float __utilization_rate))
   self)
  (:name
   (&optional __name)
   (if __name (setq _name __name)) _name)
  (:active
   (&optional (__active :null))
   (if (not (eq __active :null)) (setq _active __active)) _active)
  (:egm_convergence_met
   (&optional (__egm_convergence_met :null))
   (if (not (eq __egm_convergence_met :null)) (setq _egm_convergence_met __egm_convergence_met)) _egm_convergence_met)
  (:egm_client_state
   (&optional __egm_client_state)
   (if __egm_client_state (setq _egm_client_state __egm_client_state)) _egm_client_state)
  (:motor_state
   (&optional __motor_state)
   (if __motor_state (setq _motor_state __motor_state)) _motor_state)
  (:rapid_execution_state
   (&optional __rapid_execution_state)
   (if __rapid_execution_state (setq _rapid_execution_state __rapid_execution_state)) _rapid_execution_state)
  (:utilization_rate
   (&optional __utilization_rate)
   (if __utilization_rate (setq _utilization_rate __utilization_rate)) _utilization_rate)
  (:serialization-length
   ()
   (+
    ;; string _name
    4 (length _name)
    ;; bool _active
    1
    ;; bool _egm_convergence_met
    1
    ;; uint8 _egm_client_state
    1
    ;; uint8 _motor_state
    1
    ;; uint8 _rapid_execution_state
    1
    ;; float64 _utilization_rate
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _name
       (write-long (length _name) s) (princ _name s)
     ;; bool _active
       (if _active (write-byte -1 s) (write-byte 0 s))
     ;; bool _egm_convergence_met
       (if _egm_convergence_met (write-byte -1 s) (write-byte 0 s))
     ;; uint8 _egm_client_state
       (write-byte _egm_client_state s)
     ;; uint8 _motor_state
       (write-byte _motor_state s)
     ;; uint8 _rapid_execution_state
       (write-byte _rapid_execution_state s)
     ;; float64 _utilization_rate
       (sys::poke _utilization_rate (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _active
     (setq _active (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _egm_convergence_met
     (setq _egm_convergence_met (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; uint8 _egm_client_state
     (setq _egm_client_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _motor_state
     (setq _motor_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _rapid_execution_state
     (setq _rapid_execution_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float64 _utilization_rate
     (setq _utilization_rate (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get abb_egm_msgs::EGMChannelState :md5sum-) "bc28749df58cd58b7299c4217fd9899f")
(setf (get abb_egm_msgs::EGMChannelState :datatype-) "abb_egm_msgs/EGMChannelState")
(setf (get abb_egm_msgs::EGMChannelState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent a subset of states
#   of an EGM communication channel. I.e. a channel between an EGM client,
#   in an ABB robot controller system, and an external EGM server.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# EGM client states:
uint8 EGM_UNDEFINED = 1
uint8 EGM_ERROR     = 2
uint8 EGM_STOPPED   = 3
uint8 EGM_RUNNING   = 4

# Motor states:
uint8 MOTORS_UNDEFINED = 1
uint8 MOTORS_ON        = 2
uint8 MOTORS_OFF       = 3

# RAPID states:
uint8 RAPID_UNDEFINED = 1
uint8 RAPID_STOPPED   = 2
uint8 RAPID_RUNNING   = 3

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The EGM channel's name.
string name

# The EGM channel's active/inactive status.
bool active

# Indicator for if the EGM convergence criteria is currently fulfilled or not.
bool egm_convergence_met

# EGM client state.
uint8 egm_client_state

# Motor state.
uint8 motor_state

# RAPID execution state.
uint8 rapid_execution_state

# Utilization rate of the connected mechanical unit.
#
# An utilization rate > 100 mean that EGM motion references are
# too aggressive and should be reduced.
float64 utilization_rate

")



(provide :abb_egm_msgs/EGMChannelState "bc28749df58cd58b7299c4217fd9899f")


