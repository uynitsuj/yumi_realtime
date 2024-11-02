;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetFileContents)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetFileContents (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETFILECONTENTS")
  (make-package "ABB_ROBOT_MSGS::SETFILECONTENTS"))
(unless (find-package "ABB_ROBOT_MSGS::SETFILECONTENTSREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETFILECONTENTSREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETFILECONTENTSRESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETFILECONTENTSRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetFileContentsRequest
  :super ros::object
  :slots (_filename _contents ))

(defmethod abb_robot_msgs::SetFileContentsRequest
  (:init
   (&key
    ((:filename __filename) "")
    ((:contents __contents) "")
    )
   (send-super :init)
   (setq _filename (string __filename))
   (setq _contents (string __contents))
   self)
  (:filename
   (&optional __filename)
   (if __filename (setq _filename __filename)) _filename)
  (:contents
   (&optional __contents)
   (if __contents (setq _contents __contents)) _contents)
  (:serialization-length
   ()
   (+
    ;; string _filename
    4 (length _filename)
    ;; string _contents
    4 (length _contents)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _filename
       (write-long (length _filename) s) (princ _filename s)
     ;; string _contents
       (write-long (length _contents) s) (princ _contents s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _filename
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _filename (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _contents
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _contents (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::SetFileContentsResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetFileContentsResponse
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

(defclass abb_robot_msgs::SetFileContents
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetFileContents :md5sum-) "b9fd3989a30f4c529cc017e54ab4c80f")
(setf (get abb_robot_msgs::SetFileContents :datatype-) "abb_robot_msgs/SetFileContents")
(setf (get abb_robot_msgs::SetFileContents :request) abb_robot_msgs::SetFileContentsRequest)
(setf (get abb_robot_msgs::SetFileContents :response) abb_robot_msgs::SetFileContentsResponse)

(defmethod abb_robot_msgs::SetFileContentsRequest
  (:response () (instance abb_robot_msgs::SetFileContentsResponse :init)))

(setf (get abb_robot_msgs::SetFileContentsRequest :md5sum-) "b9fd3989a30f4c529cc017e54ab4c80f")
(setf (get abb_robot_msgs::SetFileContentsRequest :datatype-) "abb_robot_msgs/SetFileContentsRequest")
(setf (get abb_robot_msgs::SetFileContentsRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the
#   contents of a file in an ABB robot controller system.
#
# Note:
#   The file will be placed in the system's home directory. If the file
#   doesn't exist, then it will be created, otherwise it will be overwritten.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Filename (including file extension) for the file in the system.
string filename

# New contents for the file.
string contents

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

(setf (get abb_robot_msgs::SetFileContentsResponse :md5sum-) "b9fd3989a30f4c529cc017e54ab4c80f")
(setf (get abb_robot_msgs::SetFileContentsResponse :datatype-) "abb_robot_msgs/SetFileContentsResponse")
(setf (get abb_robot_msgs::SetFileContentsResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the
#   contents of a file in an ABB robot controller system.
#
# Note:
#   The file will be placed in the system's home directory. If the file
#   doesn't exist, then it will be created, otherwise it will be overwritten.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# Filename (including file extension) for the file in the system.
string filename

# New contents for the file.
string contents

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



(provide :abb_robot_msgs/SetFileContents "b9fd3989a30f4c529cc017e54ab4c80f")


