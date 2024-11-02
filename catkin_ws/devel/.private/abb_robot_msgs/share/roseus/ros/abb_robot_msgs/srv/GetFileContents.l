;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::GetFileContents)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'GetFileContents (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::GETFILECONTENTS")
  (make-package "ABB_ROBOT_MSGS::GETFILECONTENTS"))
(unless (find-package "ABB_ROBOT_MSGS::GETFILECONTENTSREQUEST")
  (make-package "ABB_ROBOT_MSGS::GETFILECONTENTSREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::GETFILECONTENTSRESPONSE")
  (make-package "ABB_ROBOT_MSGS::GETFILECONTENTSRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::GetFileContentsRequest
  :super ros::object
  :slots (_filename ))

(defmethod abb_robot_msgs::GetFileContentsRequest
  (:init
   (&key
    ((:filename __filename) "")
    )
   (send-super :init)
   (setq _filename (string __filename))
   self)
  (:filename
   (&optional __filename)
   (if __filename (setq _filename __filename)) _filename)
  (:serialization-length
   ()
   (+
    ;; string _filename
    4 (length _filename)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _filename
       (write-long (length _filename) s) (princ _filename s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _filename
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _filename (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetFileContentsResponse
  :super ros::object
  :slots (_contents _result_code _message ))

(defmethod abb_robot_msgs::GetFileContentsResponse
  (:init
   (&key
    ((:contents __contents) "")
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _contents (string __contents))
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:contents
   (&optional __contents)
   (if __contents (setq _contents __contents)) _contents)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; string _contents
    4 (length _contents)
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _contents
       (write-long (length _contents) s) (princ _contents s)
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _contents
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _contents (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetFileContents
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::GetFileContents :md5sum-) "4715c1a4564cfe2f12eb2de6ae62ccaa")
(setf (get abb_robot_msgs::GetFileContents :datatype-) "abb_robot_msgs/GetFileContents")
(setf (get abb_robot_msgs::GetFileContents :request) abb_robot_msgs::GetFileContentsRequest)
(setf (get abb_robot_msgs::GetFileContents :response) abb_robot_msgs::GetFileContentsResponse)

(defmethod abb_robot_msgs::GetFileContentsRequest
  (:response () (instance abb_robot_msgs::GetFileContentsResponse :init)))

(setf (get abb_robot_msgs::GetFileContentsRequest :md5sum-) "4715c1a4564cfe2f12eb2de6ae62ccaa")
(setf (get abb_robot_msgs::GetFileContentsRequest :datatype-) "abb_robot_msgs/GetFileContentsRequest")
(setf (get abb_robot_msgs::GetFileContentsRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the
#   contents of a file existing in an ABB robot controller system.
#
# Note:
#   The file is assumed to be located in the system's home directory.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Filename (including file extension) of the targeted file.
string filename

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Contents of the file.
string contents

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::GetFileContentsResponse :md5sum-) "4715c1a4564cfe2f12eb2de6ae62ccaa")
(setf (get abb_robot_msgs::GetFileContentsResponse :datatype-) "abb_robot_msgs/GetFileContentsResponse")
(setf (get abb_robot_msgs::GetFileContentsResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the
#   contents of a file existing in an ABB robot controller system.
#
# Note:
#   The file is assumed to be located in the system's home directory.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Filename (including file extension) of the targeted file.
string filename

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Contents of the file.
string contents

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/GetFileContents "4715c1a4564cfe2f12eb2de6ae62ccaa")


