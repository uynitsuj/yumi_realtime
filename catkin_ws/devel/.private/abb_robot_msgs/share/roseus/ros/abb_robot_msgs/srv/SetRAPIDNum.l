;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetRAPIDNum)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetRAPIDNum (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDNUM")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDNUM"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDNUMREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDNUMREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDNUMRESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDNUMRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetRAPIDNumRequest
  :super ros::object
  :slots (_path _value ))

(defmethod abb_robot_msgs::SetRAPIDNumRequest
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
    ;; float32 _value
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_robot_msgs/RAPIDSymbolPath _path
       (send _path :serialize s)
     ;; float32 _value
       (sys::poke _value (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_robot_msgs/RAPIDSymbolPath _path
     (send _path :deserialize buf ptr-) (incf ptr- (send _path :serialization-length))
   ;; float32 _value
     (setq _value (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass abb_robot_msgs::SetRAPIDNumResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetRAPIDNumResponse
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

(defclass abb_robot_msgs::SetRAPIDNum
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetRAPIDNum :md5sum-) "4108cd886758eb287b6fb706804d2b63")
(setf (get abb_robot_msgs::SetRAPIDNum :datatype-) "abb_robot_msgs/SetRAPIDNum")
(setf (get abb_robot_msgs::SetRAPIDNum :request) abb_robot_msgs::SetRAPIDNumRequest)
(setf (get abb_robot_msgs::SetRAPIDNum :response) abb_robot_msgs::SetRAPIDNumResponse)

(defmethod abb_robot_msgs::SetRAPIDNumRequest
  (:response () (instance abb_robot_msgs::SetRAPIDNumResponse :init)))

(setf (get abb_robot_msgs::SetRAPIDNumRequest :md5sum-) "4108cd886758eb287b6fb706804d2b63")
(setf (get abb_robot_msgs::SetRAPIDNumRequest :datatype-) "abb_robot_msgs/SetRAPIDNumRequest")
(setf (get abb_robot_msgs::SetRAPIDNumRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'num' symbol (e.g. a variable) defined in an ABB robot controller
#   system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
float32 value


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

(setf (get abb_robot_msgs::SetRAPIDNumResponse :md5sum-) "4108cd886758eb287b6fb706804d2b63")
(setf (get abb_robot_msgs::SetRAPIDNumResponse :datatype-) "abb_robot_msgs/SetRAPIDNumResponse")
(setf (get abb_robot_msgs::SetRAPIDNumResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'num' symbol (e.g. a variable) defined in an ABB robot controller
#   system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
float32 value


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



(provide :abb_robot_msgs/SetRAPIDNum "4108cd886758eb287b6fb706804d2b63")


