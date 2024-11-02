// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class EGMSetupSettings {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.use_filtering = null;
      this.comm_timeout = null;
    }
    else {
      if (initObj.hasOwnProperty('use_filtering')) {
        this.use_filtering = initObj.use_filtering
      }
      else {
        this.use_filtering = false;
      }
      if (initObj.hasOwnProperty('comm_timeout')) {
        this.comm_timeout = initObj.comm_timeout
      }
      else {
        this.comm_timeout = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EGMSetupSettings
    // Serialize message field [use_filtering]
    bufferOffset = _serializer.bool(obj.use_filtering, buffer, bufferOffset);
    // Serialize message field [comm_timeout]
    bufferOffset = _serializer.float32(obj.comm_timeout, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EGMSetupSettings
    let len;
    let data = new EGMSetupSettings(null);
    // Deserialize message field [use_filtering]
    data.use_filtering = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [comm_timeout]
    data.comm_timeout = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 5;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_sm_addin_msgs/EGMSetupSettings';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '13bb9a91c166ef61bfdd5d16c6d9ac65';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMSetupSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMSetupSettings" are for example used with the RAPID EGM instruction
    #   "EGMSetupUC".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Indicator for if the EGM controller should apply filtering on the
    # EGM references.
    bool use_filtering
    
    # Communication timeout [s].
    float32 comm_timeout
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EGMSetupSettings(null);
    if (msg.use_filtering !== undefined) {
      resolved.use_filtering = msg.use_filtering;
    }
    else {
      resolved.use_filtering = false
    }

    if (msg.comm_timeout !== undefined) {
      resolved.comm_timeout = msg.comm_timeout;
    }
    else {
      resolved.comm_timeout = 0.0
    }

    return resolved;
    }
};

module.exports = EGMSetupSettings;
