
(cl:in-package :asdf)

(defsystem "abb_rapid_sm_addin_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :abb_rapid_sm_addin_msgs-msg
)
  :components ((:file "_package")
    (:file "GetEGMSettings" :depends-on ("_package_GetEGMSettings"))
    (:file "_package_GetEGMSettings" :depends-on ("_package"))
    (:file "SetEGMSettings" :depends-on ("_package_SetEGMSettings"))
    (:file "_package_SetEGMSettings" :depends-on ("_package"))
    (:file "SetRAPIDRoutine" :depends-on ("_package_SetRAPIDRoutine"))
    (:file "_package_SetRAPIDRoutine" :depends-on ("_package"))
    (:file "SetSGCommand" :depends-on ("_package_SetSGCommand"))
    (:file "_package_SetSGCommand" :depends-on ("_package"))
  ))