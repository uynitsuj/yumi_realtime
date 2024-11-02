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

class RAPIDTaskState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.name = null;
      this.activated = null;
      this.execution_state = null;
      this.motion_task = null;
    }
    else {
      if (initObj.hasOwnProperty('name')) {
        this.name = initObj.name
      }
      else {
        this.name = '';
      }
      if (initObj.hasOwnProperty('activated')) {
        this.activated = initObj.activated
      }
      else {
        this.activated = false;
      }
      if (initObj.hasOwnProperty('execution_state')) {
        this.execution_state = initObj.execution_state
      }
      else {
        this.execution_state = 0;
      }
      if (initObj.hasOwnProperty('motion_task')) {
        this.motion_task = initObj.motion_task
      }
      else {
        this.motion_task = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RAPIDTaskState
    // Serialize message field [name]
    bufferOffset = _serializer.string(obj.name, buffer, bufferOffset);
    // Serialize message field [activated]
    bufferOffset = _serializer.bool(obj.activated, buffer, bufferOffset);
    // Serialize message field [execution_state]
    bufferOffset = _serializer.uint8(obj.execution_state, buffer, bufferOffset);
    // Serialize message field [motion_task]
    bufferOffset = _serializer.bool(obj.motion_task, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RAPIDTaskState
    let len;
    let data = new RAPIDTaskState(null);
    // Deserialize message field [name]
    data.name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [activated]
    data.activated = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [execution_state]
    data.execution_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [motion_task]
    data.motion_task = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.name);
    return length + 7;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_robot_msgs/RAPIDTaskState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '37a1a82a542dcaea43aef8b824e26614';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent a subset of states
    #   of a RAPID task defined in an ABB robot controller system.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Enumerations
    #-------------------------------------------------------------------------------
    # RAPID execution states:
    uint8 EXECUTION_STATE_UNKNOWN       = 1
    uint8 EXECUTION_STATE_READY         = 2
    uint8 EXECUTION_STATE_STOPPED       = 3
    uint8 EXECUTION_STATE_STARTED       = 4
    uint8 EXECUTION_STATE_UNINITIALIZED = 5
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The RAPID task's name.
    string name
    
    # The RAPID task's activated/deactivated status.
    bool activated
    
    # The RAPID task's execution state.
    uint8 execution_state
    
    # The RAPID task is a motion task.
    bool motion_task
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RAPIDTaskState(null);
    if (msg.name !== undefined) {
      resolved.name = msg.name;
    }
    else {
      resolved.name = ''
    }

    if (msg.activated !== undefined) {
      resolved.activated = msg.activated;
    }
    else {
      resolved.activated = false
    }

    if (msg.execution_state !== undefined) {
      resolved.execution_state = msg.execution_state;
    }
    else {
      resolved.execution_state = 0
    }

    if (msg.motion_task !== undefined) {
      resolved.motion_task = msg.motion_task;
    }
    else {
      resolved.motion_task = false
    }

    return resolved;
    }
};

// Constants for message
RAPIDTaskState.Constants = {
  EXECUTION_STATE_UNKNOWN: 1,
  EXECUTION_STATE_READY: 2,
  EXECUTION_STATE_STOPPED: 3,
  EXECUTION_STATE_STARTED: 4,
  EXECUTION_STATE_UNINITIALIZED: 5,
}

module.exports = RAPIDTaskState;
