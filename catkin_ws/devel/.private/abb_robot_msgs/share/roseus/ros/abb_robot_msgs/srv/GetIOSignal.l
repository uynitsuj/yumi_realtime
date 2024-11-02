;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::GetIOSignal)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'GetIOSignal (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::GETIOSIGNAL")
  (make-package "ABB_ROBOT_MSGS::GETIOSIGNAL"))
(unless (find-package "ABB_ROBOT_MSGS::GETIOSIGNALREQUEST")
  (make-package "ABB_ROBOT_MSGS::GETIOSIGNALREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::GETIOSIGNALRESPONSE")
  (make-package "ABB_ROBOT_MSGS::GETIOSIGNALRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::GetIOSignalRequest
  :super ros::object
  :slots (_signal ))

(defmethod abb_robot_msgs::GetIOSignalRequest
  (:init
   (&key
    ((:signal __signal) "")
    )
   (send-super :init)
   (setq _signal (string __signal))
   self)
  (:signal
   (&optional __signal)
   (if __signal (setq _signal __signal)) _signal)
  (:serialization-length
   ()
   (+
    ;; string _signal
    4 (length _signal)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _signal
       (write-long (length _signal) s) (princ _signal s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _signal
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _signal (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetIOSignalResponse
  :super ros::object
  :slots (_value _result_code _message ))

(defmethod abb_robot_msgs::GetIOSignalResponse
  (:init
   (&key
    ((:value __value) "")
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _value (string __value))
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:value
   (&optional __value)
   (if __value (setq _value __value)) _value)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; string _value
    4 (length _value)
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _value
       (write-long (length _value) s) (princ _value s)
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _value
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _value (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetIOSignal
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::GetIOSignal :md5sum-) "edcf90cd349492ba1ca9cc7d13c4954a")
(setf (get abb_robot_msgs::GetIOSignal :datatype-) "abb_robot_msgs/GetIOSignal")
(setf (get abb_robot_msgs::GetIOSignal :request) abb_robot_msgs::GetIOSignalRequest)
(setf (get abb_robot_msgs::GetIOSignal :response) abb_robot_msgs::GetIOSignalResponse)

(defmethod abb_robot_msgs::GetIOSignalRequest
  (:response () (instance abb_robot_msgs::GetIOSignalResponse :init)))

(setf (get abb_robot_msgs::GetIOSignalRequest :md5sum-) "edcf90cd349492ba1ca9cc7d13c4954a")
(setf (get abb_robot_msgs::GetIOSignalRequest :datatype-) "abb_robot_msgs/GetIOSignalRequest")
(setf (get abb_robot_msgs::GetIOSignalRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the
#   value of an IO-signal defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Name of the targeted IO-signal.
string signal

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Value of the IO-signal.
string value

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::GetIOSignalResponse :md5sum-) "edcf90cd349492ba1ca9cc7d13c4954a")
(setf (get abb_robot_msgs::GetIOSignalResponse :datatype-) "abb_robot_msgs/GetIOSignalResponse")
(setf (get abb_robot_msgs::GetIOSignalResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the
#   value of an IO-signal defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Name of the targeted IO-signal.
string signal

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Value of the IO-signal.
string value

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/GetIOSignal "edcf90cd349492ba1ca9cc7d13c4954a")


