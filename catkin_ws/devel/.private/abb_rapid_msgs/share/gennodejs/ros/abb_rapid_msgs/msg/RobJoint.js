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

class RobJoint {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.rax_1 = null;
      this.rax_2 = null;
      this.rax_3 = null;
      this.rax_4 = null;
      this.rax_5 = null;
      this.rax_6 = null;
    }
    else {
      if (initObj.hasOwnProperty('rax_1')) {
        this.rax_1 = initObj.rax_1
      }
      else {
        this.rax_1 = 0.0;
      }
      if (initObj.hasOwnProperty('rax_2')) {
        this.rax_2 = initObj.rax_2
      }
      else {
        this.rax_2 = 0.0;
      }
      if (initObj.hasOwnProperty('rax_3')) {
        this.rax_3 = initObj.rax_3
      }
      else {
        this.rax_3 = 0.0;
      }
      if (initObj.hasOwnProperty('rax_4')) {
        this.rax_4 = initObj.rax_4
      }
      else {
        this.rax_4 = 0.0;
      }
      if (initObj.hasOwnProperty('rax_5')) {
        this.rax_5 = initObj.rax_5
      }
      else {
        this.rax_5 = 0.0;
      }
      if (initObj.hasOwnProperty('rax_6')) {
        this.rax_6 = initObj.rax_6
      }
      else {
        this.rax_6 = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RobJoint
    // Serialize message field [rax_1]
    bufferOffset = _serializer.float32(obj.rax_1, buffer, bufferOffset);
    // Serialize message field [rax_2]
    bufferOffset = _serializer.float32(obj.rax_2, buffer, bufferOffset);
    // Serialize message field [rax_3]
    bufferOffset = _serializer.float32(obj.rax_3, buffer, bufferOffset);
    // Serialize message field [rax_4]
    bufferOffset = _serializer.float32(obj.rax_4, buffer, bufferOffset);
    // Serialize message field [rax_5]
    bufferOffset = _serializer.float32(obj.rax_5, buffer, bufferOffset);
    // Serialize message field [rax_6]
    bufferOffset = _serializer.float32(obj.rax_6, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RobJoint
    let len;
    let data = new RobJoint(null);
    // Deserialize message field [rax_1]
    data.rax_1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rax_2]
    data.rax_2 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rax_3]
    data.rax_3 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rax_4]
    data.rax_4 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rax_5]
    data.rax_5 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rax_6]
    data.rax_6 = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/RobJoint';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '24a88117c4b385974e50d44f6663eb2a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "robjoint" (joint position of robot axes).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The position of robot axis 1-6 in degrees from the calibration position.
    float32 rax_1
    float32 rax_2
    float32 rax_3
    float32 rax_4
    float32 rax_5
    float32 rax_6
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RobJoint(null);
    if (msg.rax_1 !== undefined) {
      resolved.rax_1 = msg.rax_1;
    }
    else {
      resolved.rax_1 = 0.0
    }

    if (msg.rax_2 !== undefined) {
      resolved.rax_2 = msg.rax_2;
    }
    else {
      resolved.rax_2 = 0.0
    }

    if (msg.rax_3 !== undefined) {
      resolved.rax_3 = msg.rax_3;
    }
    else {
      resolved.rax_3 = 0.0
    }

    if (msg.rax_4 !== undefined) {
      resolved.rax_4 = msg.rax_4;
    }
    else {
      resolved.rax_4 = 0.0
    }

    if (msg.rax_5 !== undefined) {
      resolved.rax_5 = msg.rax_5;
    }
    else {
      resolved.rax_5 = 0.0
    }

    if (msg.rax_6 !== undefined) {
      resolved.rax_6 = msg.rax_6;
    }
    else {
      resolved.rax_6 = 0.0
    }

    return resolved;
    }
};

module.exports = RobJoint;
