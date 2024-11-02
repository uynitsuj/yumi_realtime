// Auto-generated. Do not edit!

// (in-package abb_egm_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let EGMChannelState = require('./EGMChannelState.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class EGMState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.egm_channels = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('egm_channels')) {
        this.egm_channels = initObj.egm_channels
      }
      else {
        this.egm_channels = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EGMState
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [egm_channels]
    // Serialize the length for message field [egm_channels]
    bufferOffset = _serializer.uint32(obj.egm_channels.length, buffer, bufferOffset);
    obj.egm_channels.forEach((val) => {
      bufferOffset = EGMChannelState.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EGMState
    let len;
    let data = new EGMState(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [egm_channels]
    // Deserialize array length for message field [egm_channels]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.egm_channels = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.egm_channels[i] = EGMChannelState.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.egm_channels.forEach((val) => {
      length += EGMChannelState.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_egm_msgs/EGMState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a239c0f02980c16cd34f9e56fbc03ff7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent a subset of states
    #   of all (known) EGM communication channels.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Metadata.
    std_msgs/Header header
    
    # EGM channel states.
    EGMChannelState[] egm_channels
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: abb_egm_msgs/EGMChannelState
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
    const resolved = new EGMState(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.egm_channels !== undefined) {
      resolved.egm_channels = new Array(msg.egm_channels.length);
      for (let i = 0; i < resolved.egm_channels.length; ++i) {
        resolved.egm_channels[i] = EGMChannelState.Resolve(msg.egm_channels[i]);
      }
    }
    else {
      resolved.egm_channels = []
    }

    return resolved;
    }
};

module.exports = EGMState;
