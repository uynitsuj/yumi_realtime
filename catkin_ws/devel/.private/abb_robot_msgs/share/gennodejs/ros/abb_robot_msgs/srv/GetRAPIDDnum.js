// Auto-generated. Do not edit!

// (in-package abb_robot_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let RAPIDSymbolPath = require('../msg/RAPIDSymbolPath.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class GetRAPIDDnumRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.path = null;
    }
    else {
      if (initObj.hasOwnProperty('path')) {
        this.path = initObj.path
      }
      else {
        this.path = new RAPIDSymbolPath();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetRAPIDDnumRequest
    // Serialize message field [path]
    bufferOffset = RAPIDSymbolPath.serialize(obj.path, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetRAPIDDnumRequest
    let len;
    let data = new GetRAPIDDnumRequest(null);
    // Deserialize message field [path]
    data.path = RAPIDSymbolPath.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += RAPIDSymbolPath.getMessageSize(object.path);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/GetRAPIDDnumRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '575c1566b746b035b579d998cd699eb8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to define a way to get the value
    #   of a RAPID 'dnum' symbol (e.g. a variable) defined in an ABB robot
    #   controller system.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Service request fields
    #-------------------------------------------------------------------------------
    # Path to the targeted RAPID symbol.
    RAPIDSymbolPath path
    
    
    ================================================================================
    MSG: abb_robot_msgs/RAPIDSymbolPath
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
    const resolved = new GetRAPIDDnumRequest(null);
    if (msg.path !== undefined) {
      resolved.path = RAPIDSymbolPath.Resolve(msg.path)
    }
    else {
      resolved.path = new RAPIDSymbolPath()
    }

    return resolved;
    }
};

class GetRAPIDDnumResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.value = null;
      this.result_code = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('value')) {
        this.value = initObj.value
      }
      else {
        this.value = 0.0;
      }
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
    // Serializes a message object of type GetRAPIDDnumResponse
    // Serialize message field [value]
    bufferOffset = _serializer.float64(obj.value, buffer, bufferOffset);
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetRAPIDDnumResponse
    let len;
    let data = new GetRAPIDDnumResponse(null);
    // Deserialize message field [value]
    data.value = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [result_code]
    data.result_code = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.message);
    return length + 14;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/GetRAPIDDnumResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e8223c10294ae2ec72c03c61fc136636';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    #-------------------------------------------------------------------------------
    # Service response fields
    #-------------------------------------------------------------------------------
    # Value of the RAPID symbol.
    float64 value
    
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
    const resolved = new GetRAPIDDnumResponse(null);
    if (msg.value !== undefined) {
      resolved.value = msg.value;
    }
    else {
      resolved.value = 0.0
    }

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
  Request: GetRAPIDDnumRequest,
  Response: GetRAPIDDnumResponse,
  md5sum() { return '8bea96ce1c5c1ccf9562161041904147'; },
  datatype() { return 'abb_robot_msgs/GetRAPIDDnum'; }
};
