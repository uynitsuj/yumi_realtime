;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetRAPIDString)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetRAPIDString (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDSTRING")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDSTRING"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDSTRINGREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDSTRINGREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDSTRINGRESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDSTRINGRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetRAPIDStringRequest
  :super ros::object
  :slots (_path _value ))

(defmethod abb_robot_msgs::SetRAPIDStringRequest
  (:init
   (&key
    ((:path __path) (instance abb_robot_msgs::RAPIDSymbolPath :init))
    ((:value __value) "")
    )
   (send-super :init)
   (setq _path __path)
   (setq _value (string __value))
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
    ;; string _value
    4 (length _value)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_robot_msgs/RAPIDSymbolPath _path
       (send _path :serialize s)
     ;; string _value
       (write-long (length _value) s) (princ _value s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_robot_msgs/RAPIDSymbolPath _path
     (send _path :deserialize buf ptr-) (incf ptr- (send _path :serialization-length))
   ;; string _value
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _value (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::SetRAPIDStringResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetRAPIDStringResponse
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

(defclass abb_robot_msgs::SetRAPIDString
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetRAPIDString :md5sum-) "ca1f63f0ec40ce75b501bc490a118a6c")
(setf (get abb_robot_msgs::SetRAPIDString :datatype-) "abb_robot_msgs/SetRAPIDString")
(setf (get abb_robot_msgs::SetRAPIDString :request) abb_robot_msgs::SetRAPIDStringRequest)
(setf (get abb_robot_msgs::SetRAPIDString :response) abb_robot_msgs::SetRAPIDStringResponse)

(defmethod abb_robot_msgs::SetRAPIDStringRequest
  (:response () (instance abb_robot_msgs::SetRAPIDStringResponse :init)))

(setf (get abb_robot_msgs::SetRAPIDStringRequest :md5sum-) "ca1f63f0ec40ce75b501bc490a118a6c")
(setf (get abb_robot_msgs::SetRAPIDStringRequest :datatype-) "abb_robot_msgs/SetRAPIDStringRequest")
(setf (get abb_robot_msgs::SetRAPIDStringRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'string' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
string value


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

(setf (get abb_robot_msgs::SetRAPIDStringResponse :md5sum-) "ca1f63f0ec40ce75b501bc490a118a6c")
(setf (get abb_robot_msgs::SetRAPIDStringResponse :datatype-) "abb_robot_msgs/SetRAPIDStringResponse")
(setf (get abb_robot_msgs::SetRAPIDStringResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'string' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
string value


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



(provide :abb_robot_msgs/SetRAPIDString "ca1f63f0ec40ce75b501bc490a118a6c")


