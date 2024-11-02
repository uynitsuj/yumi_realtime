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

class TriggerWithResultCodeRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TriggerWithResultCodeRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TriggerWithResultCodeRequest
    let len;
    let data = new TriggerWithResultCodeRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/TriggerWithResultCodeRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to allow triggering a service
    #   where the response includes both a numerical result code and a human
    #   readable message.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Service request fields (none)
    #-------------------------------------------------------------------------------
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TriggerWithResultCodeRequest(null);
    return resolved;
    }
};

class TriggerWithResultCodeResponse {
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
    // Serializes a message object of type TriggerWithResultCodeResponse
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TriggerWithResultCodeResponse
    let len;
    let data = new TriggerWithResultCodeResponse(null);
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
    return 'abb_robot_msgs/TriggerWithResultCodeResponse';
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
    
    # Informational, e.g. for error messages
    string message
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TriggerWithResultCodeResponse(null);
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
  Request: TriggerWithResultCodeRequest,
  Response: TriggerWithResultCodeResponse,
  md5sum() { return 'aab0cc4b756aaf08eb67ad6459b36b27'; },
  datatype() { return 'abb_robot_msgs/TriggerWithResultCode'; }
};
