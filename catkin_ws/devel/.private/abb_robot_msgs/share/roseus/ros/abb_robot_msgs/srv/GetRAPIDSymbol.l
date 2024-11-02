;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::GetRAPIDSymbol)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'GetRAPIDSymbol (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::GETRAPIDSYMBOL")
  (make-package "ABB_ROBOT_MSGS::GETRAPIDSYMBOL"))
(unless (find-package "ABB_ROBOT_MSGS::GETRAPIDSYMBOLREQUEST")
  (make-package "ABB_ROBOT_MSGS::GETRAPIDSYMBOLREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::GETRAPIDSYMBOLRESPONSE")
  (make-package "ABB_ROBOT_MSGS::GETRAPIDSYMBOLRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::GetRAPIDSymbolRequest
  :super ros::object
  :slots (_path ))

(defmethod abb_robot_msgs::GetRAPIDSymbolRequest
  (:init
   (&key
    ((:path __path) (instance abb_robot_msgs::RAPIDSymbolPath :init))
    )
   (send-super :init)
   (setq _path __path)
   self)
  (:path
   (&rest __path)
   (if (keywordp (car __path))
       (send* _path __path)
     (progn
       (if __path (setq _path (car __path)))
       _path)))
  (:serialization-length
   ()
   (+
    ;; abb_robot_msgs/RAPIDSymbolPath _path
    (send _path :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_robot_msgs/RAPIDSymbolPath _path
       (send _path :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_robot_msgs/RAPIDSymbolPath _path
     (send _path :deserialize buf ptr-) (incf ptr- (send _path :serialization-length))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetRAPIDSymbolResponse
  :super ros::object
  :slots (_value _result_code _message ))

(defmethod abb_robot_msgs::GetRAPIDSymbolResponse
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

(defclass abb_robot_msgs::GetRAPIDSymbol
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::GetRAPIDSymbol :md5sum-) "bde39a2871b891ab4e4dba6d1b2afbe4")
(setf (get abb_robot_msgs::GetRAPIDSymbol :datatype-) "abb_robot_msgs/GetRAPIDSymbol")
(setf (get abb_robot_msgs::GetRAPIDSymbol :request) abb_robot_msgs::GetRAPIDSymbolRequest)
(setf (get abb_robot_msgs::GetRAPIDSymbol :response) abb_robot_msgs::GetRAPIDSymbolResponse)

(defmethod abb_robot_msgs::GetRAPIDSymbolRequest
  (:response () (instance abb_robot_msgs::GetRAPIDSymbolResponse :init)))

(setf (get abb_robot_msgs::GetRAPIDSymbolRequest :md5sum-) "bde39a2871b891ab4e4dba6d1b2afbe4")
(setf (get abb_robot_msgs::GetRAPIDSymbolRequest :datatype-) "abb_robot_msgs/GetRAPIDSymbolRequest")
(setf (get abb_robot_msgs::GetRAPIDSymbolRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the value
#   of a RAPID symbol (e.g. a variable) defined in an ABB robot controller
#   system.
#
# Note:
#   The RAPID symbol's value will be returned in raw text format.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path


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
# Value (in raw text format) of the RAPID symbol.
string value

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::GetRAPIDSymbolResponse :md5sum-) "bde39a2871b891ab4e4dba6d1b2afbe4")
(setf (get abb_robot_msgs::GetRAPIDSymbolResponse :datatype-) "abb_robot_msgs/GetRAPIDSymbolResponse")
(setf (get abb_robot_msgs::GetRAPIDSymbolResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the value
#   of a RAPID symbol (e.g. a variable) defined in an ABB robot controller
#   system.
#
# Note:
#   The RAPID symbol's value will be returned in raw text format.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path


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
# Value (in raw text format) of the RAPID symbol.
string value

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/GetRAPIDSymbol "bde39a2871b891ab4e4dba6d1b2afbe4")


