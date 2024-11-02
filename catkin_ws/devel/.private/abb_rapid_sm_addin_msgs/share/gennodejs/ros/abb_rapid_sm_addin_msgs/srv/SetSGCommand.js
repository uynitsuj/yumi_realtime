// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetSGCommandRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.task = null;
      this.command = null;
      this.target_position = null;
    }
    else {
      if (initObj.hasOwnProperty('task')) {
        this.task = initObj.task
      }
      else {
        this.task = '';
      }
      if (initObj.hasOwnProperty('command')) {
        this.command = initObj.command
      }
      else {
        this.command = 0;
      }
      if (initObj.hasOwnProperty('target_position')) {
        this.target_position = initObj.target_position
      }
      else {
        this.target_position = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetSGCommandRequest
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    // Serialize message field [command]
    bufferOffset = _serializer.uint8(obj.command, buffer, bufferOffset);
    // Serialize message field [target_position]
    bufferOffset = _serializer.float32(obj.target_position, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSGCommandRequest
    let len;
    let data = new SetSGCommandRequest(null);
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [command]
    data.command = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [target_position]
    data.target_position = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.task);
    return length + 9;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_rapid_sm_addin_msgs/SetSGCommandRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8840583d66e469cae970ed301faf98c0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to define a way to set the values
    #   of predefined RAPID variables for commanding a SmartGripper tool.
    #
    # Note:
    #   These variables are defined in each RAPID motion task, in ABB robot
    #   controller systems, that are using the RobotWare "StateMachine Add-In"
    #   together with a SmartGripper tool.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Enumerations
    #-------------------------------------------------------------------------------
    # Supported "StateMachine Add-In" SmartGripper commands:
    #
    # Note: These enumeration values differ from the RAPID implementation
    #       values to avoid conflict with default values in ROS messages.
    uint8 SG_COMMAND_UNKNOWN      = 1
    uint8 SG_COMMAND_NONE         = 2
    uint8 SG_COMMAND_INITIALIZE   = 3
    uint8 SG_COMMAND_CALIBRATE    = 4
    uint8 SG_COMMAND_MOVE_TO      = 5
    uint8 SG_COMMAND_GRIP_IN      = 6
    uint8 SG_COMMAND_GRIP_OUT     = 7
    uint8 SG_COMMAND_BLOW_ON_1    = 8
    uint8 SG_COMMAND_BLOW_ON_2    = 9
    uint8 SG_COMMAND_BLOW_OFF_1   = 10
    uint8 SG_COMMAND_BLOW_OFF_2   = 11
    uint8 SG_COMMAND_VACUUM_ON_1  = 12
    uint8 SG_COMMAND_VACUUM_ON_2  = 13
    uint8 SG_COMMAND_VACUUM_OFF_1 = 14
    uint8 SG_COMMAND_VACUUM_OFF_2 = 15
    
    #-------------------------------------------------------------------------------
    # Service request fields
    #-------------------------------------------------------------------------------
    # The RAPID task that runs the "StateMachine Add-In" instance.
    string task
    
    # Desired SmartGripper command.
    uint8 command
    
    # Desired target position [mm] (only used for the 'SG_COMMAND_MOVE_TO' command).
    float32 target_position
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetSGCommandRequest(null);
    if (msg.task !== undefined) {
      resolved.task = msg.task;
    }
    else {
      resolved.task = ''
    }

    if (msg.command !== undefined) {
      resolved.command = msg.command;
    }
    else {
      resolved.command = 0
    }

    if (msg.target_position !== undefined) {
      resolved.target_position = msg.target_position;
    }
    else {
      resolved.target_position = 0.0
    }

    return resolved;
    }
};

// Constants for message
SetSGCommandRequest.Constants = {
  SG_COMMAND_UNKNOWN: 1,
  SG_COMMAND_NONE: 2,
  SG_COMMAND_INITIALIZE: 3,
  SG_COMMAND_CALIBRATE: 4,
  SG_COMMAND_MOVE_TO: 5,
  SG_COMMAND_GRIP_IN: 6,
  SG_COMMAND_GRIP_OUT: 7,
  SG_COMMAND_BLOW_ON_1: 8,
  SG_COMMAND_BLOW_ON_2: 9,
  SG_COMMAND_BLOW_OFF_1: 10,
  SG_COMMAND_BLOW_OFF_2: 11,
  SG_COMMAND_VACUUM_ON_1: 12,
  SG_COMMAND_VACUUM_ON_2: 13,
  SG_COMMAND_VACUUM_OFF_1: 14,
  SG_COMMAND_VACUUM_OFF_2: 15,
}

class SetSGCommandResponse {
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
    // Serializes a message object of type SetSGCommandResponse
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSGCommandResponse
    let len;
    let data = new SetSGCommandResponse(null);
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
    return 'abb_rapid_sm_addin_msgs/SetSGCommandResponse';
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
    const resolved = new SetSGCommandResponse(null);
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
  Request: SetSGCommandRequest,
  Response: SetSGCommandResponse,
  md5sum() { return '7aa352af5c8c7b889375c50673d12253'; },
  datatype() { return 'abb_rapid_sm_addin_msgs/SetSGCommand'; }
};
