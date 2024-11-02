;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::SetRAPIDRoutine)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'SetRAPIDRoutine (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETRAPIDROUTINE")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETRAPIDROUTINE"))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETRAPIDROUTINEREQUEST")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETRAPIDROUTINEREQUEST"))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETRAPIDROUTINERESPONSE")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETRAPIDROUTINERESPONSE"))

(in-package "ROS")





(defclass abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest
  :super ros::object
  :slots (_task _routine ))

(defmethod abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest
  (:init
   (&key
    ((:task __task) "")
    ((:routine __routine) "")
    )
   (send-super :init)
   (setq _task (string __task))
   (setq _routine (string __routine))
   self)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:routine
   (&optional __routine)
   (if __routine (setq _routine __routine)) _routine)
  (:serialization-length
   ()
   (+
    ;; string _task
    4 (length _task)
    ;; string _routine
    4 (length _routine)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; string _routine
       (write-long (length _routine) s) (princ _routine s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _routine
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _routine (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse
  (:init
   (&key
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_rapid_sm_addin_msgs::SetRAPIDRoutine
  :super ros::object
  :slots ())

(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutine :md5sum-) "652cc4087927fd3e9d05a7309ce84794")
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutine :datatype-) "abb_rapid_sm_addin_msgs/SetRAPIDRoutine")
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutine :request) abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest)
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutine :response) abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse)

(defmethod abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest
  (:response () (instance abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse :init)))

(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest :md5sum-) "652cc4087927fd3e9d05a7309ce84794")
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest :datatype-) "abb_rapid_sm_addin_msgs/SetRAPIDRoutineRequest")
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutineRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a predefined RAPID variable for specifying a custom RAPID routine to run.
#
# Note:
#   This variable is defined in each RAPID motion task, in ABB robot
#   controller systems, that are using the RobotWare \"StateMachine Add-In\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# The RAPID task that runs the \"StateMachine Add-In\" instance.
string task

# Desired custom RAPID routine to run.
string routine

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse :md5sum-) "652cc4087927fd3e9d05a7309ce84794")
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse :datatype-) "abb_rapid_sm_addin_msgs/SetRAPIDRoutineResponse")
(setf (get abb_rapid_sm_addin_msgs::SetRAPIDRoutineResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a predefined RAPID variable for specifying a custom RAPID routine to run.
#
# Note:
#   This variable is defined in each RAPID motion task, in ABB robot
#   controller systems, that are using the RobotWare \"StateMachine Add-In\".
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# The RAPID task that runs the \"StateMachine Add-In\" instance.
string task

# Desired custom RAPID routine to run.
string routine

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_rapid_sm_addin_msgs/SetRAPIDRoutine "652cc4087927fd3e9d05a7309ce84794")


