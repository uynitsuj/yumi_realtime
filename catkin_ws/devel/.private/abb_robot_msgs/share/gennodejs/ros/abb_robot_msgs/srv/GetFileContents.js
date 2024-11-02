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

class GetFileContentsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.filename = null;
    }
    else {
      if (initObj.hasOwnProperty('filename')) {
        this.filename = initObj.filename
      }
      else {
        this.filename = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetFileContentsRequest
    // Serialize message field [filename]
    bufferOffset = _serializer.string(obj.filename, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetFileContentsRequest
    let len;
    let data = new GetFileContentsRequest(null);
    // Deserialize message field [filename]
    data.filename = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.filename);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/GetFileContentsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '030824f52a0628ead956fb9d67e66ae9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to define a way to get the
    #   contents of a file existing in an ABB robot controller system.
    #
    # Note:
    #   The file is assumed to be located in the system's home directory.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Service request fields
    #-------------------------------------------------------------------------------
    # Filename (including file extension) of the targeted file.
    string filename
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetFileContentsRequest(null);
    if (msg.filename !== undefined) {
      resolved.filename = msg.filename;
    }
    else {
      resolved.filename = ''
    }

    return resolved;
    }
};

class GetFileContentsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.contents = null;
      this.result_code = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('contents')) {
        this.contents = initObj.contents
      }
      else {
        this.contents = '';
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
    // Serializes a message object of type GetFileContentsResponse
    // Serialize message field [contents]
    bufferOffset = _serializer.string(obj.contents, buffer, bufferOffset);
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetFileContentsResponse
    let len;
    let data = new GetFileContentsResponse(null);
    // Deserialize message field [contents]
    data.contents = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [result_code]
    data.result_code = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.contents);
    length += _getByteLength(object.message);
    return length + 10;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_robot_msgs/GetFileContentsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3a83ecd2e6d8d82acbb6145f3b481cff';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    #-------------------------------------------------------------------------------
    # Service response fields
    #-------------------------------------------------------------------------------
    # Contents of the file.
    string contents
    
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
    const resolved = new GetFileContentsResponse(null);
    if (msg.contents !== undefined) {
      resolved.contents = msg.contents;
    }
    else {
      resolved.contents = ''
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
  Request: GetFileContentsRequest,
  Response: GetFileContentsResponse,
  md5sum() { return '4715c1a4564cfe2f12eb2de6ae62ccaa'; },
  datatype() { return 'abb_robot_msgs/GetFileContents'; }
};
