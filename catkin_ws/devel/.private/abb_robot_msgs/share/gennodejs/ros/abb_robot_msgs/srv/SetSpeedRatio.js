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

class SetSpeedRatioRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.speed_ratio = null;
    }
    else {
      if (initObj.hasOwnProperty('speed_ratio')) {
        this.speed_ratio = initObj.speed_ratio
      }
      else {
        this.speed_ratio = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetSpeedRatioRequest
    // Serialize message field [speed_ratio]
    bufferOffset = _serializer.uint8(obj.speed_ratio, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSpeedRatioRequest
    let len;
    let data = new SetSpeedRatioRequest(null);
    // Deserialize message field [speed_ratio]
    data.speed_ratio = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/SetSpeedRatioRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '433e0c752185d553ba16644b83665ddc';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to define a way to set the
    #   speed ratio for an ABB robot controller system's RAPID motions
    #   (e.g. MoveAbsJ and MoveL).
    #
    # Note:
    #   The speed ratio must be an integer in the range [0, 100] (i.e. inclusive).
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Service request fields
    #-------------------------------------------------------------------------------
    # New speed ratio.
    uint8 speed_ratio
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetSpeedRatioRequest(null);
    if (msg.speed_ratio !== undefined) {
      resolved.speed_ratio = msg.speed_ratio;
    }
    else {
      resolved.speed_ratio = 0
    }

    return resolved;
    }
};

class SetSpeedRatioResponse {
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
    // Serializes a message object of type SetSpeedRatioResponse
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSpeedRatioResponse
    let len;
    let data = new SetSpeedRatioResponse(null);
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
    return 'abb_robot_msgs/SetSpeedRatioResponse';
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
    const resolved = new SetSpeedRatioResponse(null);
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
  Request: SetSpeedRatioRequest,
  Response: SetSpeedRatioResponse,
  md5sum() { return 'a37a98881c87fe9569621a0efb03c471'; },
  datatype() { return 'abb_robot_msgs/SetSpeedRatio'; }
};
