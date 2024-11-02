// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let abb_rapid_msgs = _finder('abb_rapid_msgs');

//-----------------------------------------------------------

class EGMRunSettings {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.cond_time = null;
      this.ramp_in_time = null;
      this.offset = null;
      this.pos_corr_gain = null;
    }
    else {
      if (initObj.hasOwnProperty('cond_time')) {
        this.cond_time = initObj.cond_time
      }
      else {
        this.cond_time = 0.0;
      }
      if (initObj.hasOwnProperty('ramp_in_time')) {
        this.ramp_in_time = initObj.ramp_in_time
      }
      else {
        this.ramp_in_time = 0.0;
      }
      if (initObj.hasOwnProperty('offset')) {
        this.offset = initObj.offset
      }
      else {
        this.offset = new abb_rapid_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('pos_corr_gain')) {
        this.pos_corr_gain = initObj.pos_corr_gain
      }
      else {
        this.pos_corr_gain = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EGMRunSettings
    // Serialize message field [cond_time]
    bufferOffset = _serializer.float32(obj.cond_time, buffer, bufferOffset);
    // Serialize message field [ramp_in_time]
    bufferOffset = _serializer.float32(obj.ramp_in_time, buffer, bufferOffset);
    // Serialize message field [offset]
    bufferOffset = abb_rapid_msgs.msg.Pose.serialize(obj.offset, buffer, bufferOffset);
    // Serialize message field [pos_corr_gain]
    bufferOffset = _serializer.float32(obj.pos_corr_gain, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EGMRunSettings
    let len;
    let data = new EGMRunSettings(null);
    // Deserialize message field [cond_time]
    data.cond_time = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [ramp_in_time]
    data.ramp_in_time = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [offset]
    data.offset = abb_rapid_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [pos_corr_gain]
    data.pos_corr_gain = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 40;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_sm_addin_msgs/EGMRunSettings';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f6a2d3b6d7d6ffc880c4bc86612a6691';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMRunSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMRunSettings" are for example used with the RAPID EGM instructions
    #   "EGMRunJoint" and "EGMRunPose".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Condition time [s] for how long an EGM motion's convergence criteria must be
    # fulfilled before the motion is considered to be finished.
    float32 cond_time
    
    # Ramp in time [s].
    float32 ramp_in_time
    
    # Static offset applied on top of the EGM references
    # (only used in EGM pose mode).
    abb_rapid_msgs/Pose offset
    
    # Position correction gain for the EGM controller.
    float32 pos_corr_gain
    
    ================================================================================
    MSG: abb_rapid_msgs/Pose
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "pose" (coordinate transformations).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The displacement in position (x, y, and z) of the coordinate system.
    Pos trans
    
    # The rotation of the coordinate system.
    Orient rot
    
    ================================================================================
    MSG: abb_rapid_msgs/Pos
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "pos" (positions (only X, Y and Z)).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The x-, y- and z-value of the position.
    float32 x
    float32 y
    float32 z
    
    ================================================================================
    MSG: abb_rapid_msgs/Orient
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "orient" (orientation).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Quaternion 1-4.
    float32 q1
    float32 q2
    float32 q3
    float32 q4
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EGMRunSettings(null);
    if (msg.cond_time !== undefined) {
      resolved.cond_time = msg.cond_time;
    }
    else {
      resolved.cond_time = 0.0
    }

    if (msg.ramp_in_time !== undefined) {
      resolved.ramp_in_time = msg.ramp_in_time;
    }
    else {
      resolved.ramp_in_time = 0.0
    }

    if (msg.offset !== undefined) {
      resolved.offset = abb_rapid_msgs.msg.Pose.Resolve(msg.offset)
    }
    else {
      resolved.offset = new abb_rapid_msgs.msg.Pose()
    }

    if (msg.pos_corr_gain !== undefined) {
      resolved.pos_corr_gain = msg.pos_corr_gain;
    }
    else {
      resolved.pos_corr_gain = 0.0
    }

    return resolved;
    }
};

module.exports = EGMRunSettings;
