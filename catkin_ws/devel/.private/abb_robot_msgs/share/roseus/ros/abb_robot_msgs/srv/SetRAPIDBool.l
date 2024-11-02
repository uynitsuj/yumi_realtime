;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetRAPIDBool)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetRAPIDBool (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDBOOL")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDBOOL"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDBOOLREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDBOOLREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETRAPIDBOOLRESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETRAPIDBOOLRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetRAPIDBoolRequest
  :super ros::object
  :slots (_path _value ))

(defmethod abb_robot_msgs::SetRAPIDBoolRequest
  (:init
   (&key
    ((:path __path) (instance abb_robot_msgs::RAPIDSymbolPath :init))
    ((:value __value) nil)
    )
   (send-super :init)
   (setq _path __path)
   (setq _value __value)
   self)
  (:path
   (&rest __path)
   (if (keywordp (car __path))
       (send* _path __path)
     (progn
       (if __path (setq _path (car __path)))
       _path)))
  (:value
   (&optional (__value :null))
   (if (not (eq __value :null)) (setq _value __value)) _value)
  (:serialization-length
   ()
   (+
    ;; abb_robot_msgs/RAPIDSymbolPath _path
    (send _path :serialization-length)
    ;; bool _value
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; abb_robot_msgs/RAPIDSymbolPath _path
       (send _path :serialize s)
     ;; bool _value
       (if _value (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; abb_robot_msgs/RAPIDSymbolPath _path
     (send _path :deserialize buf ptr-) (incf ptr- (send _path :serialization-length))
   ;; bool _value
     (setq _value (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass abb_robot_msgs::SetRAPIDBoolResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetRAPIDBoolResponse
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

(defclass abb_robot_msgs::SetRAPIDBool
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetRAPIDBool :md5sum-) "932d876676bb80924a225c457e4d06a3")
(setf (get abb_robot_msgs::SetRAPIDBool :datatype-) "abb_robot_msgs/SetRAPIDBool")
(setf (get abb_robot_msgs::SetRAPIDBool :request) abb_robot_msgs::SetRAPIDBoolRequest)
(setf (get abb_robot_msgs::SetRAPIDBool :response) abb_robot_msgs::SetRAPIDBoolResponse)

(defmethod abb_robot_msgs::SetRAPIDBoolRequest
  (:response () (instance abb_robot_msgs::SetRAPIDBoolResponse :init)))

(setf (get abb_robot_msgs::SetRAPIDBoolRequest :md5sum-) "932d876676bb80924a225c457e4d06a3")
(setf (get abb_robot_msgs::SetRAPIDBoolRequest :datatype-) "abb_robot_msgs/SetRAPIDBoolRequest")
(setf (get abb_robot_msgs::SetRAPIDBoolRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'bool' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
bool value


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

(setf (get abb_robot_msgs::SetRAPIDBoolResponse :md5sum-) "932d876676bb80924a225c457e4d06a3")
(setf (get abb_robot_msgs::SetRAPIDBoolResponse :datatype-) "abb_robot_msgs/SetRAPIDBoolResponse")
(setf (get abb_robot_msgs::SetRAPIDBoolResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the value
#   of a RAPID 'bool' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Path to the targeted RAPID symbol.
RAPIDSymbolPath path

# New value of the RAPID symbol.
bool value


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



(provide :abb_robot_msgs/SetRAPIDBool "932d876676bb80924a225c457e4d06a3")


