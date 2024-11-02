;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::GetRAPIDBool)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'GetRAPIDBool (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::GETRAPIDBOOL")
  (make-package "ABB_ROBOT_MSGS::GETRAPIDBOOL"))
(unless (find-package "ABB_ROBOT_MSGS::GETRAPIDBOOLREQUEST")
  (make-package "ABB_ROBOT_MSGS::GETRAPIDBOOLREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::GETRAPIDBOOLRESPONSE")
  (make-package "ABB_ROBOT_MSGS::GETRAPIDBOOLRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::GetRAPIDBoolRequest
  :super ros::object
  :slots (_path ))

(defmethod abb_robot_msgs::GetRAPIDBoolRequest
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

(defclass abb_robot_msgs::GetRAPIDBoolResponse
  :super ros::object
  :slots (_value _result_code _message ))

(defmethod abb_robot_msgs::GetRAPIDBoolResponse
  (:init
   (&key
    ((:value __value) nil)
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _value __value)
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:value
   (&optional (__value :null))
   (if (not (eq __value :null)) (setq _value __value)) _value)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; bool _value
    1
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _value
       (if _value (write-byte -1 s) (write-byte 0 s))
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _value
     (setq _value (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetRAPIDBool
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::GetRAPIDBool :md5sum-) "54173522f636a2206a3c2982067e45e8")
(setf (get abb_robot_msgs::GetRAPIDBool :datatype-) "abb_robot_msgs/GetRAPIDBool")
(setf (get abb_robot_msgs::GetRAPIDBool :request) abb_robot_msgs::GetRAPIDBoolRequest)
(setf (get abb_robot_msgs::GetRAPIDBool :response) abb_robot_msgs::GetRAPIDBoolResponse)

(defmethod abb_robot_msgs::GetRAPIDBoolRequest
  (:response () (instance abb_robot_msgs::GetRAPIDBoolResponse :init)))

(setf (get abb_robot_msgs::GetRAPIDBoolRequest :md5sum-) "54173522f636a2206a3c2982067e45e8")
(setf (get abb_robot_msgs::GetRAPIDBoolRequest :datatype-) "abb_robot_msgs/GetRAPIDBoolRequest")
(setf (get abb_robot_msgs::GetRAPIDBoolRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the value
#   of a RAPID 'bool' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
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
# Value of the RAPID symbol.
bool value

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::GetRAPIDBoolResponse :md5sum-) "54173522f636a2206a3c2982067e45e8")
(setf (get abb_robot_msgs::GetRAPIDBoolResponse :datatype-) "abb_robot_msgs/GetRAPIDBoolResponse")
(setf (get abb_robot_msgs::GetRAPIDBoolResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the value
#   of a RAPID 'bool' symbol (e.g. a variable) defined in an ABB robot
#   controller system.
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
# Value of the RAPID symbol.
bool value

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/GetRAPIDBool "54173522f636a2206a3c2982067e45e8")


