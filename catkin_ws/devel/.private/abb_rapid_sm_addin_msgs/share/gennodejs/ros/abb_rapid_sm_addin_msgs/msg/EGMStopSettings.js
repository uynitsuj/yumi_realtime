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

class EGMStopSettings {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.ramp_out_time = null;
    }
    else {
      if (initObj.hasOwnProperty('ramp_out_time')) {
        this.ramp_out_time = initObj.ramp_out_time
      }
      else {
        this.ramp_out_time = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EGMStopSettings
    // Serialize message field [ramp_out_time]
    bufferOffset = _serializer.float32(obj.ramp_out_time, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EGMStopSettings
    let len;
    let data = new EGMStopSettings(null);
    // Deserialize message field [ramp_out_time]
    data.ramp_out_time = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_sm_addin_msgs/EGMStopSettings';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0e3716aa777ece435e2f7343649fe19b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMStopSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMStopSettings" are for example used with the RAPID EGM instruction
    #   "EGMStop".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Ramp out time [s].
    float32 ramp_out_time
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EGMStopSettings(null);
    if (msg.ramp_out_time !== undefined) {
      resolved.ramp_out_time = msg.ramp_out_time;
    }
    else {
      resolved.ramp_out_time = 0.0
    }

    return resolved;
    }
};

module.exports = EGMStopSettings;
