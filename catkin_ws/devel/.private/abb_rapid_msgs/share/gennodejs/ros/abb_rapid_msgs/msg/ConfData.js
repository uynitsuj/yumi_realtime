// Auto-generated. Do not edit!

// (in-package abb_rapid_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class ConfData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.cf1 = null;
      this.cf4 = null;
      this.cf6 = null;
      this.cfx = null;
    }
    else {
      if (initObj.hasOwnProperty('cf1')) {
        this.cf1 = initObj.cf1
      }
      else {
        this.cf1 = 0.0;
      }
      if (initObj.hasOwnProperty('cf4')) {
        this.cf4 = initObj.cf4
      }
      else {
        this.cf4 = 0.0;
      }
      if (initObj.hasOwnProperty('cf6')) {
        this.cf6 = initObj.cf6
      }
      else {
        this.cf6 = 0.0;
      }
      if (initObj.hasOwnProperty('cfx')) {
        this.cfx = initObj.cfx
      }
      else {
        this.cfx = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ConfData
    // Serialize message field [cf1]
    bufferOffset = _serializer.float32(obj.cf1, buffer, bufferOffset);
    // Serialize message field [cf4]
    bufferOffset = _serializer.float32(obj.cf4, buffer, bufferOffset);
    // Serialize message field [cf6]
    bufferOffset = _serializer.float32(obj.cf6, buffer, bufferOffset);
    // Serialize message field [cfx]
    bufferOffset = _serializer.float32(obj.cfx, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ConfData
    let len;
    let data = new ConfData(null);
    // Deserialize message field [cf1]
    data.cf1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cf4]
    data.cf4 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cf6]
    data.cf6 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cfx]
    data.cfx = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/ConfData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '21f1129f94cd63dd0cdf33be2bddc75e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "confdata" (robot configuration data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Rotating axis:
    # - The current quadrant of axis 1, 4 and 6,
    #   expressed as a positive or negative integer.
    # Linear axis:
    # - The current meter interval of axis 1, 4 and 6,
    #   expressed as a positive or negative integer.
    float32 cf1
    float32 cf4
    float32 cf6
    
    # Rotating axis:
    # - For serial link robots, the current robot configuration,
    #   expressed as an integer in the range from 0 to 7.
    # - For SCARA robots, the current robot configuration,
    #   expressed as an integer in the range from 0 to 1.
    # - For 7-axis robots, the current robot configuration,
    #   expressed as an integer in the range from 0 to 7.
    # - For paint robots, the current quadrant of axis 5,
    #   expressed as a positive or negative integer.
    # - For other robots, using the current quadrant of axis 2,
    #   expressed as a positive or negative integer.
    # Linear axis:
    # - The current meter interval of axis 2,
    #   expressed as a positive or negative integer.
    float32 cfx
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ConfData(null);
    if (msg.cf1 !== undefined) {
      resolved.cf1 = msg.cf1;
    }
    else {
      resolved.cf1 = 0.0
    }

    if (msg.cf4 !== undefined) {
      resolved.cf4 = msg.cf4;
    }
    else {
      resolved.cf4 = 0.0
    }

    if (msg.cf6 !== undefined) {
      resolved.cf6 = msg.cf6;
    }
    else {
      resolved.cf6 = 0.0
    }

    if (msg.cfx !== undefined) {
      resolved.cfx = msg.cfx;
    }
    else {
      resolved.cfx = 0.0
    }

    return resolved;
    }
};

module.exports = ConfData;
