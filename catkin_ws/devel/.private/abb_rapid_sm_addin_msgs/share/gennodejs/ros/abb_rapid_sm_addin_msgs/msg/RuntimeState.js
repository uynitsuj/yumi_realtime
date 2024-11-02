// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let StateMachineState = require('./StateMachineState.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class RuntimeState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.state_machines = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('state_machines')) {
        this.state_machines = initObj.state_machines
      }
      else {
        this.state_machines = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RuntimeState
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [state_machines]
    // Serialize the length for message field [state_machines]
    bufferOffset = _serializer.uint32(obj.state_machines.length, buffer, bufferOffset);
    obj.state_machines.forEach((val) => {
      bufferOffset = StateMachineState.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RuntimeState
    let len;
    let data = new RuntimeState(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [state_machines]
    // Deserialize array length for message field [state_machines]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.state_machines = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.state_machines[i] = StateMachineState.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.state_machines.forEach((val) => {
      length += StateMachineState.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_sm_addin_msgs/RuntimeState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1f803b084f80e27cb1a45c04dc77c6e7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the runtime states
    #   of all (known) RobotWare "StateMachine Add-In" RAPID instances.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Metadata.
    std_msgs/Header header
    
    # "StateMachine Add-In" instances.
    StateMachineState[] state_machines
    
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
    MSG: abb_rapid_sm_addin_msgs/StateMachineState
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
    const resolved = new RuntimeState(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.state_machines !== undefined) {
      resolved.state_machines = new Array(msg.state_machines.length);
      for (let i = 0; i < resolved.state_machines.length; ++i) {
        resolved.state_machines[i] = StateMachineState.Resolve(msg.state_machines[i]);
      }
    }
    else {
      resolved.state_machines = []
    }

    return resolved;
    }
};

module.exports = RuntimeState;
