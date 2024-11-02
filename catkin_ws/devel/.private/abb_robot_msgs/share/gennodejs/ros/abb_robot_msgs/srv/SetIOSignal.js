// Auto-generated. Do not edit!

// (in-package abb_robot_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetIOSignalRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.signal = null;
      this.value = null;
    }
    else {
      if (initObj.hasOwnProperty('signal')) {
        this.signal = initObj.signal
      }
      else {
        this.signal = '';
      }
      if (initObj.hasOwnProperty('value')) {
        this.value = initObj.value
      }
      else {
        this.value = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetIOSignalRequest
    // Serialize message field [signal]
    bufferOffset = _serializer.string(obj.signal, buffer, bufferOffset);
    // Serialize message field [value]
    bufferOffset = _serializer.string(obj.value, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetIOSignalRequest
    let len;
    let data = new SetIOSignalRequest(null);
    // Deserialize message field [signal]
    data.signal = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [value]
    data.value = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.signal);
    length += _getByteLength(object.value);
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/SetIOSignalRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '17cb31e0dfb7106dce1621cc1fa3f3bf';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to define a way to set the
    #   value of an IO-signal defined in an ABB robot controller system.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Service request fields
    #-------------------------------------------------------------------------------
    # Name of the targeted IO-signal.
    string signal
    
    # New value for the IO-signal.
    string value
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetIOSignalRequest(null);
    if (msg.signal !== undefined) {
      resolved.signal = msg.signal;
    }
    else {
      resolved.signal = ''
    }

    if (msg.value !== undefined) {
      resolved.value = msg.value;
    }
    else {
      resolved.value = ''
    }

    return resolved;
    }
};

class SetIOSignalResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.result_code = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('result_code')) {
        this.result_code = initObj.result_code
      }
      else {
        this.result_code = 0;
      }
      if (initObj.hasOwnProperty('message')) {
        this.message = initObj.message
      }
      else {
        this.message = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetIOSignalResponse
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetIOSignalResponse
    let len;
    let data = new SetIOSignalResponse(null);
    // Deserialize message field [result_code]
    data.result_code = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.message);
    return length + 6;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/SetIOSignalResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'aab0cc4b756aaf08eb67ad6459b36b27';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    #-------------------------------------------------------------------------------
    # Service response fields
    #-------------------------------------------------------------------------------
    # Service success/failure indicator.
    # Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
    uint16 result_code
    
    # Status message (empty if service succeeded).
    string message
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetIOSignalResponse(null);
    if (msg.result_code !== undefined) {
      resolved.result_code = msg.result_code;
    }
    else {
      resolved.result_code = 0
    }

    if (msg.message !== undefined) {
      resolved.message = msg.message;
    }
    else {
      resolved.message = ''
    }

    return resolved;
    }
};

module.exports = {
  Request: SetIOSignalRequest,
  Response: SetIOSignalResponse,
  md5sum() { return '0950ef5b96964231922a8ac9fd2a1318'; },
  datatype() { return 'abb_robot_msgs/SetIOSignal'; }
};
