// Auto-generated. Do not edit!

// (in-package abb_robot_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class RAPIDSymbolPath {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.task = null;
      this.module = null;
      this.symbol = null;
    }
    else {
      if (initObj.hasOwnProperty('task')) {
        this.task = initObj.task
      }
      else {
        this.task = '';
      }
      if (initObj.hasOwnProperty('module')) {
        this.module = initObj.module
      }
      else {
        this.module = '';
      }
      if (initObj.hasOwnProperty('symbol')) {
        this.symbol = initObj.symbol
      }
      else {
        this.symbol = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RAPIDSymbolPath
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    // Serialize message field [module]
    bufferOffset = _serializer.string(obj.module, buffer, bufferOffset);
    // Serialize message field [symbol]
    bufferOffset = _serializer.string(obj.symbol, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RAPIDSymbolPath
    let len;
    let data = new RAPIDSymbolPath(null);
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [module]
    data.module = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [symbol]
    data.symbol = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.task);
    length += _getByteLength(object.module);
    length += _getByteLength(object.symbol);
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_robot_msgs/RAPIDSymbolPath';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3214a5a49942fc04a524a160d9874980';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RAPIDSymbolPath(null);
    if (msg.task !== undefined) {
      resolved.task = msg.task;
    }
    else {
      resolved.task = ''
    }

    if (msg.module !== undefined) {
      resolved.module = msg.module;
    }
    else {
      resolved.module = ''
    }

    if (msg.symbol !== undefined) {
      resolved.symbol = msg.symbol;
    }
    else {
      resolved.symbol = ''
    }

    return resolved;
    }
};

module.exports = RAPIDSymbolPath;
