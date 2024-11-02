// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class StateMachineState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.rapid_task = null;
      this.sm_state = null;
      this.egm_action = null;
    }
    else {
      if (initObj.hasOwnProperty('rapid_task')) {
        this.rapid_task = initObj.rapid_task
      }
      else {
        this.rapid_task = '';
      }
      if (initObj.hasOwnProperty('sm_state')) {
        this.sm_state = initObj.sm_state
      }
      else {
        this.sm_state = 0;
      }
      if (initObj.hasOwnProperty('egm_action')) {
        this.egm_action = initObj.egm_action
      }
      else {
        this.egm_action = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type StateMachineState
    // Serialize message field [rapid_task]
    bufferOffset = _serializer.string(obj.rapid_task, buffer, bufferOffset);
    // Serialize message field [sm_state]
    bufferOffset = _serializer.uint8(obj.sm_state, buffer, bufferOffset);
    // Serialize message field [egm_action]
    bufferOffset = _serializer.uint8(obj.egm_action, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type StateMachineState
    let len;
    let data = new StateMachineState(null);
    // Deserialize message field [rapid_task]
    data.rapid_task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [sm_state]
    data.sm_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [egm_action]
    data.egm_action = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.rapid_task);
    return length + 6;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_sm_addin_msgs/StateMachineState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'de6354e383149183de9c15ab3d871c7b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the state of an
    #   instance of the RobotWare "StateMachine Add-In"'s RAPID implementation.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Enumerations
    #-------------------------------------------------------------------------------
    # Possible "StateMachine Add-In" RAPID states:
    #
    # Note: These enumeration values differ from the RAPID implementation
    #       values to avoid conflict with default values in ROS messages.
    uint8 SM_STATE_UNKNOWN           = 1
    uint8 SM_STATE_IDLE              = 2
    uint8 SM_STATE_INITIALIZE        = 3
    uint8 SM_STATE_RUN_RAPID_ROUTINE = 4
    uint8 SM_STATE_RUN_EGM_ROUTINE   = 5
    
    # Possible "StateMachine Add-In" RAPID EGM actions:
    #
    # Note: These enumeration values differ from the RAPID implementation
    #       values to avoid conflict with default values in ROS messages.
    uint8 EGM_ACTION_UNKNOWN      = 1
    uint8 EGM_ACTION_NONE         = 2
    uint8 EGM_ACTION_RUN_JOINT    = 3
    uint8 EGM_ACTION_RUN_POSE     = 4
    uint8 EGM_ACTION_STOP         = 5
    uint8 EGM_ACTION_START_STREAM = 6
    uint8 EGM_ACTION_STOP_STREAM  = 7
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The RAPID task that runs the "StateMachine Add-In" instance.
    string rapid_task
    
    # The "StateMachine Add-In" instance's current state.
    uint8 sm_state
    
    # The "StateMachine Add-In" instance's current EGM action
    # (only used if the RobotWare EGM option is present).
    uint8 egm_action
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new StateMachineState(null);
    if (msg.rapid_task !== undefined) {
      resolved.rapid_task = msg.rapid_task;
    }
    else {
      resolved.rapid_task = ''
    }

    if (msg.sm_state !== undefined) {
      resolved.sm_state = msg.sm_state;
    }
    else {
      resolved.sm_state = 0
    }

    if (msg.egm_action !== undefined) {
      resolved.egm_action = msg.egm_action;
    }
    else {
      resolved.egm_action = 0
    }

    return resolved;
    }
};

// Constants for message
StateMachineState.Constants = {
  SM_STATE_UNKNOWN: 1,
  SM_STATE_IDLE: 2,
  SM_STATE_INITIALIZE: 3,
  SM_STATE_RUN_RAPID_ROUTINE: 4,
  SM_STATE_RUN_EGM_ROUTINE: 5,
  EGM_ACTION_UNKNOWN: 1,
  EGM_ACTION_NONE: 2,
  EGM_ACTION_RUN_JOINT: 3,
  EGM_ACTION_RUN_POSE: 4,
  EGM_ACTION_STOP: 5,
  EGM_ACTION_START_STREAM: 6,
  EGM_ACTION_STOP_STREAM: 7,
}

module.exports = StateMachineState;
