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

class ExtJoint {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.eax_a = null;
      this.eax_b = null;
      this.eax_c = null;
      this.eax_d = null;
      this.eax_e = null;
      this.eax_f = null;
    }
    else {
      if (initObj.hasOwnProperty('eax_a')) {
        this.eax_a = initObj.eax_a
      }
      else {
        this.eax_a = 0.0;
      }
      if (initObj.hasOwnProperty('eax_b')) {
        this.eax_b = initObj.eax_b
      }
      else {
        this.eax_b = 0.0;
      }
      if (initObj.hasOwnProperty('eax_c')) {
        this.eax_c = initObj.eax_c
      }
      else {
        this.eax_c = 0.0;
      }
      if (initObj.hasOwnProperty('eax_d')) {
        this.eax_d = initObj.eax_d
      }
      else {
        this.eax_d = 0.0;
      }
      if (initObj.hasOwnProperty('eax_e')) {
        this.eax_e = initObj.eax_e
      }
      else {
        this.eax_e = 0.0;
      }
      if (initObj.hasOwnProperty('eax_f')) {
        this.eax_f = initObj.eax_f
      }
      else {
        this.eax_f = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ExtJoint
    // Serialize message field [eax_a]
    bufferOffset = _serializer.float32(obj.eax_a, buffer, bufferOffset);
    // Serialize message field [eax_b]
    bufferOffset = _serializer.float32(obj.eax_b, buffer, bufferOffset);
    // Serialize message field [eax_c]
    bufferOffset = _serializer.float32(obj.eax_c, buffer, bufferOffset);
    // Serialize message field [eax_d]
    bufferOffset = _serializer.float32(obj.eax_d, buffer, bufferOffset);
    // Serialize message field [eax_e]
    bufferOffset = _serializer.float32(obj.eax_e, buffer, bufferOffset);
    // Serialize message field [eax_f]
    bufferOffset = _serializer.float32(obj.eax_f, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ExtJoint
    let len;
    let data = new ExtJoint(null);
    // Deserialize message field [eax_a]
    data.eax_a = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [eax_b]
    data.eax_b = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [eax_c]
    data.eax_c = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [eax_d]
    data.eax_d = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [eax_e]
    data.eax_e = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [eax_f]
    data.eax_f = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/ExtJoint';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f8550d9e82bcfae84c73de38fde07b03';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "extjoint" (position of external joints).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The position of the external logical axis "a"-"f",
    # expressed in degrees or mm (depending on the type of axis).
    float32 eax_a
    float32 eax_b
    float32 eax_c
    float32 eax_d
    float32 eax_e
    float32 eax_f
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ExtJoint(null);
    if (msg.eax_a !== undefined) {
      resolved.eax_a = msg.eax_a;
    }
    else {
      resolved.eax_a = 0.0
    }

    if (msg.eax_b !== undefined) {
      resolved.eax_b = msg.eax_b;
    }
    else {
      resolved.eax_b = 0.0
    }

    if (msg.eax_c !== undefined) {
      resolved.eax_c = msg.eax_c;
    }
    else {
      resolved.eax_c = 0.0
    }

    if (msg.eax_d !== undefined) {
      resolved.eax_d = msg.eax_d;
    }
    else {
      resolved.eax_d = 0.0
    }

    if (msg.eax_e !== undefined) {
      resolved.eax_e = msg.eax_e;
    }
    else {
      resolved.eax_e = 0.0
    }

    if (msg.eax_f !== undefined) {
      resolved.eax_f = msg.eax_f;
    }
    else {
      resolved.eax_f = 0.0
    }

    return resolved;
    }
};

module.exports = ExtJoint;
