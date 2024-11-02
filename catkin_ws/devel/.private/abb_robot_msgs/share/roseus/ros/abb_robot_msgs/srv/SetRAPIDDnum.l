;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetRAPIDDnum)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetRAPIDDnum (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDDNUM")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDDNUM"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDDNUMREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDDNUMREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDDNUMRESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDDNUMRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetRAPIDDnumRequest
  :super ros::object
  :slots (_path _value ))

(defmethod abb_robot_msgs::SetRAPIDDnumRequest
  (:init
   (&key
    ((:path __path) (instance abb_robot_msgs::RAPIDSymbolPath :init))
    ((:value __value) 0.0)
    )
   (send-super :init)
   (setq _path __path)
   (setq _value (float __value))
   self)
  (:path
   (&rest __path)
   (if (keywordp (car __path))
       (send* _path __path)
     (progn
       (if __path (setq _path (car __path)))
       _path)))
  (:value
   (&optional __value)
   (if __value (setq _value __value)) _value)
  (:serialization-length
   ()
   (+
    ;; abb_robot_msgs/RAPIDSymbolPath _path
    (send _path :serialization-length)
    ;; float64 _value
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_robot_msgs/RAPIDSymbolPath _path
       (send _path :serialize s)
     ;; float64 _value
       (sys::poke _value (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_robot_msgs/RAPIDSymbolPath _path
     (send _path :deserialize buf ptr-) (incf ptr- (send _path :serialization-length))
   ;; float64 _value
     (setq _value (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass abb_robot_msgs::SetRAPIDDnumResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetRAPIDDnumResponse
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

(defclass abb_robot_msgs::SetRAPIDDnum
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetRAPIDDnum :md5sum-) "62c05fe9bd5c47b90203ea370499cfc5")
(setf (get abb_robot_msgs::SetRAPIDDnum :datatype-) "abb_robot_msgs/SetRAPIDDnum")
(setf (get abb_robot_msgs::SetRAPIDDnum :request) abb_robot_msgs::SetRAPIDDnumRequest)
(setf (get abb_robot_msgs::SetRAPIDDnum :response) abb_robot_msgs::SetRAPIDDnumResponse)

(defmethod abb_robot_msgs::SetRAPIDDnumRequest
  (:response () (instance abb_robot_msgs::SetRAPIDDnumResponse :init)))

(setf (get abb_robot_msgs::SetRAPIDDnumRequest :md5sum-) "62c05fe9bd5c47b90203ea370499cfc5")
(setf (get abb_robot_msgs::SetRAPIDDnumRequest :datatype-) "abb_robot_msgs/SetRAPIDDnumRequest")
(setf (get abb_robot_msgs::SetRAPIDDnumRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'dnum' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
float64 value


================================================================================
MSG: abb_robot_msgs/RAPIDSymbolPath
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the path to a RAPID
#   symbol (e.g. a variable) defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).
string task

# Name of the RAPID module, where the symbol has been defined.
string module

# Name of the RAPID symbol in question.
string symbol
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

(setf (get abb_robot_msgs::SetRAPIDDnumResponse :md5sum-) "62c05fe9bd5c47b90203ea370499cfc5")
(setf (get abb_robot_msgs::SetRAPIDDnumResponse :datatype-) "abb_robot_msgs/SetRAPIDDnumResponse")
(setf (get abb_robot_msgs::SetRAPIDDnumResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'dnum' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
float64 value


================================================================================
MSG: abb_robot_msgs/RAPIDSymbolPath
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the path to a RAPID
#   symbol (e.g. a variable) defined in an ABB robot controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Name of the RAPID task, where the symbol exists (i.e. the symbol's context).
string task

# Name of the RAPID module, where the symbol has been defined.
string module

# Name of the RAPID symbol in question.
string symbol
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



(provide :abb_robot_msgs/SetRAPIDDnum "62c05fe9bd5c47b90203ea370499cfc5")


