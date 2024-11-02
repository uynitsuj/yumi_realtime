;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::RAPIDSymbolPath)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'RAPIDSymbolPath (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::RAPIDSYMBOLPATH")
  (make-package "ABB_ROBOT_MSGS::RAPIDSYMBOLPATH"))

(in-package "ROS")
;;//! \htmlinclude RAPIDSymbolPath.msg.html


(defclass abb_robot_msgs::RAPIDSymbolPath
  :super ros::object
  :slots (_task _module _symbol ))

(defmethod abb_robot_msgs::RAPIDSymbolPath
  (:init
   (&key
    ((:task __task) "")
    ((:module __module) "")
    ((:symbol __symbol) "")
    )
   (send-super :init)
   (setq _task (string __task))
   (setq _module (string __module))
   (setq _symbol (string __symbol))
   self)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:module
   (&optional __module)
   (if __module (setq _module __module)) _module)
  (:symbol
   (&optional __symbol)
   (if __symbol (setq _symbol __symbol)) _symbol)
  (:serialization-length
   ()
   (+
    ;; string _task
    4 (length _task)
    ;; string _module
    4 (length _module)
    ;; string _symbol
    4 (length _symbol)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; string _module
       (write-long (length _module) s) (princ _module s)
     ;; string _symbol
       (write-long (length _symbol) s) (princ _symbol s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _module
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _module (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _symbol
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _symbol (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get abb_robot_msgs::RAPIDSymbolPath :md5sum-) "3214a5a49942fc04a524a160d9874980")
(setf (get abb_robot_msgs::RAPIDSymbolPath :datatype-) "abb_robot_msgs/RAPIDSymbolPath")
(setf (get abb_robot_msgs::RAPIDSymbolPath :definition-)
      "#-------------------------------------------------------------------------------
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

")



(provide :abb_robot_msgs/RAPIDSymbolPath "3214a5a49942fc04a524a160d9874980")


