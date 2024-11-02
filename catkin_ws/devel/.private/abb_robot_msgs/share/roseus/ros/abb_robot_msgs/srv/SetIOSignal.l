;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetIOSignal)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetIOSignal (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETIOSIGNAL")
  (make-package "ABB_ROBOT_MSGS::SETIOSIGNAL"))
(unless (find-package "ABB_ROBOT_MSGS::SETIOSIGNALREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETIOSIGNALREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETIOSIGNALRESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETIOSIGNALRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetIOSignalRequest
  :super ros::object
  :slots (_signal _value ))

(defmethod abb_robot_msgs::SetIOSignalRequest
  (:init
   (&key
    ((:signal __signal) "")
    ((:value __value) "")
    )
   (send-super :init)
   (setq _signal (string __signal))
   (setq _value (string __value))
   self)
  (:signal
   (&optional __signal)
   (if __signal (setq _signal __signal)) _signal)
  (:value
   (&optional __value)
   (if __value (setq _value __value)) _value)
  (:serialization-length
   ()
   (+
    ;; string _signal
    4 (length _signal)
    ;; string _value
    4 (length _value)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _signal
       (write-long (length _signal) s) (princ _signal s)
     ;; string _value
       (write-long (length _value) s) (princ _value s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _signal
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _signal (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _value
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _value (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::SetIOSignalResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetIOSignalResponse
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

(defclass abb_robot_msgs::SetIOSignal
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetIOSignal :md5sum-) "0950ef5b96964231922a8ac9fd2a1318")
(setf (get abb_robot_msgs::SetIOSignal :datatype-) "abb_robot_msgs/SetIOSignal")
(setf (get abb_robot_msgs::SetIOSignal :request) abb_robot_msgs::SetIOSignalRequest)
(setf (get abb_robot_msgs::SetIOSignal :response) abb_robot_msgs::SetIOSignalResponse)

(defmethod abb_robot_msgs::SetIOSignalRequest
  (:response () (instance abb_robot_msgs::SetIOSignalResponse :init)))

(setf (get abb_robot_msgs::SetIOSignalRequest :md5sum-) "0950ef5b96964231922a8ac9fd2a1318")
(setf (get abb_robot_msgs::SetIOSignalRequest :datatype-) "abb_robot_msgs/SetIOSignalRequest")
(setf (get abb_robot_msgs::SetIOSignalRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the
#   value of an IO-signal defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Name of the targeted IO-signal.
string signal

# New value for the IO-signal.
string value

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

(setf (get abb_robot_msgs::SetIOSignalResponse :md5sum-) "0950ef5b96964231922a8ac9fd2a1318")
(setf (get abb_robot_msgs::SetIOSignalResponse :datatype-) "abb_robot_msgs/SetIOSignalResponse")
(setf (get abb_robot_msgs::SetIOSignalResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the
#   value of an IO-signal defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Name of the targeted IO-signal.
string signal

# New value for the IO-signal.
string value

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



(provide :abb_robot_msgs/SetIOSignal "0950ef5b96964231922a8ac9fd2a1318")


