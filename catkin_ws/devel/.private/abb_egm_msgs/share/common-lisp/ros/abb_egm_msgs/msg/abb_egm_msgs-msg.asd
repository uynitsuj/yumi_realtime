
(cl:in-package :asdf)

(defsystem "abb_egm_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "EGMChannelState" :depends-on ("_package_EGMChannelState"))
    (:file "_package_EGMChannelState" :depends-on ("_package"))
    (:file "EGMState" :depends-on ("_package_EGMState"))
    (:file "_package_EGMState" :depends-on ("_package"))
  ))