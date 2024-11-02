// Auto-generated. Do not edit!

// (in-package abb_egm_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class EGMChannelState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.name = null;
      this.active = null;
      this.egm_convergence_met = null;
      this.egm_client_state = null;
      this.motor_state = null;
      this.rapid_execution_state = null;
      this.utilization_rate = null;
    }
    else {
      if (initObj.hasOwnProperty('name')) {
        this.name = initObj.name
      }
      else {
        this.name = '';
      }
      if (initObj.hasOwnProperty('active')) {
        this.active = initObj.active
      }
      else {
        this.active = false;
      }
      if (initObj.hasOwnProperty('egm_convergence_met')) {
        this.egm_convergence_met = initObj.egm_convergence_met
      }
      else {
        this.egm_convergence_met = false;
      }
      if (initObj.hasOwnProperty('egm_client_state')) {
        this.egm_client_state = initObj.egm_client_state
      }
      else {
        this.egm_client_state = 0;
      }
      if (initObj.hasOwnProperty('motor_state')) {
        this.motor_state = initObj.motor_state
      }
      else {
        this.motor_state = 0;
      }
      if (initObj.hasOwnProperty('rapid_execution_state')) {
        this.rapid_execution_state = initObj.rapid_execution_state
      }
      else {
        this.rapid_execution_state = 0;
      }
      if (initObj.hasOwnProperty('utilization_rate')) {
        this.utilization_rate = initObj.utilization_rate
      }
      else {
        this.utilization_rate = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EGMChannelState
    // Serialize message field [name]
    bufferOffset = _serializer.string(obj.name, buffer, bufferOffset);
    // Serialize message field [active]
    bufferOffset = _serializer.bool(obj.active, buffer, bufferOffset);
    // Serialize message field [egm_convergence_met]
    bufferOffset = _serializer.bool(obj.egm_convergence_met, buffer, bufferOffset);
    // Serialize message field [egm_client_state]
    bufferOffset = _serializer.uint8(obj.egm_client_state, buffer, bufferOffset);
    // Serialize message field [motor_state]
    bufferOffset = _serializer.uint8(obj.motor_state, buffer, bufferOffset);
    // Serialize message field [rapid_execution_state]
    bufferOffset = _serializer.uint8(obj.rapid_execution_state, buffer, bufferOffset);
    // Serialize message field [utilization_rate]
    bufferOffset = _serializer.float64(obj.utilization_rate, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EGMChannelState
    let len;
    let data = new EGMChannelState(null);
    // Deserialize message field [name]
    data.name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [active]
    data.active = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [egm_convergence_met]
    data.egm_convergence_met = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [egm_client_state]
    data.egm_client_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [motor_state]
    data.motor_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [rapid_execution_state]
    data.rapid_execution_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [utilization_rate]
    data.utilization_rate = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.name);
    return length + 17;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_egm_msgs/EGMChannelState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'bc28749df58cd58b7299c4217fd9899f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent a subset of states
    #   of an EGM communication channel. I.e. a channel between an EGM client,
    #   in an ABB robot controller system, and an external EGM server.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Enumerations
    #-------------------------------------------------------------------------------
    # EGM client states:
    uint8 EGM_UNDEFINED = 1
    uint8 EGM_ERROR     = 2
    uint8 EGM_STOPPED   = 3
    uint8 EGM_RUNNING   = 4
    
    # Motor states:
    uint8 MOTORS_UNDEFINED = 1
    uint8 MOTORS_ON        = 2
    uint8 MOTORS_OFF       = 3
    
    # RAPID states:
    uint8 RAPID_UNDEFINED = 1
    uint8 RAPID_STOPPED   = 2
    uint8 RAPID_RUNNING   = 3
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The EGM channel's name.
    string name
    
    # The EGM channel's active/inactive status.
    bool active
    
    # Indicator for if the EGM convergence criteria is currently fulfilled or not.
    bool egm_convergence_met
    
    # EGM client state.
    uint8 egm_client_state
    
    # Motor state.
    uint8 motor_state
    
    # RAPID execution state.
    uint8 rapid_execution_state
    
    # Utilization rate of the connected mechanical unit.
    #
    # An utilization rate > 100 mean that EGM motion references are
    # too aggressive and should be reduced.
    float64 utilization_rate
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EGMChannelState(null);
    if (msg.name !== undefined) {
      resolved.name = msg.name;
    }
    else {
      resolved.name = ''
    }

    if (msg.active !== undefined) {
      resolved.active = msg.active;
    }
    else {
      resolved.active = false
    }

    if (msg.egm_convergence_met !== undefined) {
      resolved.egm_convergence_met = msg.egm_convergence_met;
    }
    else {
      resolved.egm_convergence_met = false
    }

    if (msg.egm_client_state !== undefined) {
      resolved.egm_client_state = msg.egm_client_state;
    }
    else {
      resolved.egm_client_state = 0
    }

    if (msg.motor_state !== undefined) {
      resolved.motor_state = msg.motor_state;
    }
    else {
      resolved.motor_state = 0
    }

    if (msg.rapid_execution_state !== undefined) {
      resolved.rapid_execution_state = msg.rapid_execution_state;
    }
    else {
      resolved.rapid_execution_state = 0
    }

    if (msg.utilization_rate !== undefined) {
      resolved.utilization_rate = msg.utilization_rate;
    }
    else {
      resolved.utilization_rate = 0.0
    }

    return resolved;
    }
};

// Constants for message
EGMChannelState.Constants = {
  EGM_UNDEFINED: 1,
  EGM_ERROR: 2,
  EGM_STOPPED: 3,
  EGM_RUNNING: 4,
  MOTORS_UNDEFINED: 1,
  MOTORS_ON: 2,
  MOTORS_OFF: 3,
  RAPID_UNDEFINED: 1,
  RAPID_STOPPED: 2,
  RAPID_RUNNING: 3,
}

module.exports = EGMChannelState;
