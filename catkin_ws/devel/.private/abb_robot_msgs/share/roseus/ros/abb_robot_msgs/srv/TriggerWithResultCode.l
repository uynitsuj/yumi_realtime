;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::TriggerWithResultCode)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'TriggerWithResultCode (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::TRIGGERWITHRESULTCODE")
  (make-package "ABB_ROBOT_MSGS::TRIGGERWITHRESULTCODE"))
(unless (find-package "ABB_ROBOT_MSGS::TRIGGERWITHRESULTCODEREQUEST")
  (make-package "ABB_ROBOT_MSGS::TRIGGERWITHRESULTCODEREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::TRIGGERWITHRESULTCODERESPONSE")
  (make-package "ABB_ROBOT_MSGS::TRIGGERWITHRESULTCODERESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::TriggerWithResultCodeRequest
  :super ros::object
  :slots ())

(defmethod abb_robot_msgs::TriggerWithResultCodeRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass abb_robot_msgs::TriggerWithResultCodeResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::TriggerWithResultCodeResponse
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

(defclass abb_robot_msgs::TriggerWithResultCode
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::TriggerWithResultCode :md5sum-) "aab0cc4b756aaf08eb67ad6459b36b27")
(setf (get abb_robot_msgs::TriggerWithResultCode :datatype-) "abb_robot_msgs/TriggerWithResultCode")
(setf (get abb_robot_msgs::TriggerWithResultCode :request) abb_robot_msgs::TriggerWithResultCodeRequest)
(setf (get abb_robot_msgs::TriggerWithResultCode :response) abb_robot_msgs::TriggerWithResultCodeResponse)

(defmethod abb_robot_msgs::TriggerWithResultCodeRequest
  (:response () (instance abb_robot_msgs::TriggerWithResultCodeResponse :init)))

(setf (get abb_robot_msgs::TriggerWithResultCodeRequest :md5sum-) "aab0cc4b756aaf08eb67ad6459b36b27")
(setf (get abb_robot_msgs::TriggerWithResultCodeRequest :datatype-) "abb_robot_msgs/TriggerWithResultCodeRequest")
(setf (get abb_robot_msgs::TriggerWithResultCodeRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to allow triggering a service
#   where the response includes both a numerical result code and a human
#   readable message.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields (none)
#-------------------------------------------------------------------------------

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Informational, e.g. for error messages
string message

")

(setf (get abb_robot_msgs::TriggerWithResultCodeResponse :md5sum-) "aab0cc4b756aaf08eb67ad6459b36b27")
(setf (get abb_robot_msgs::TriggerWithResultCodeResponse :datatype-) "abb_robot_msgs/TriggerWithResultCodeResponse")
(setf (get abb_robot_msgs::TriggerWithResultCodeResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to allow triggering a service
#   where the response includes both a numerical result code and a human
#   readable message.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields (none)
#-------------------------------------------------------------------------------

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Informational, e.g. for error messages
string message

")



(provide :abb_robot_msgs/TriggerWithResultCode "aab0cc4b756aaf08eb67ad6459b36b27")


