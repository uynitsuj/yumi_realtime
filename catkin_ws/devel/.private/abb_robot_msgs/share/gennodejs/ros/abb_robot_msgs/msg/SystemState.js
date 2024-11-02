// Auto-generated. Do not edit!

// (in-package abb_robot_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let RAPIDTaskState = require('./RAPIDTaskState.js');
let MechanicalUnitState = require('./MechanicalUnitState.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class SystemState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.motors_on = null;
      this.auto_mode = null;
      this.rapid_running = null;
      this.rapid_tasks = null;
      this.mechanical_units = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('motors_on')) {
        this.motors_on = initObj.motors_on
      }
      else {
        this.motors_on = false;
      }
      if (initObj.hasOwnProperty('auto_mode')) {
        this.auto_mode = initObj.auto_mode
      }
      else {
        this.auto_mode = false;
      }
      if (initObj.hasOwnProperty('rapid_running')) {
        this.rapid_running = initObj.rapid_running
      }
      else {
        this.rapid_running = false;
      }
      if (initObj.hasOwnProperty('rapid_tasks')) {
        this.rapid_tasks = initObj.rapid_tasks
      }
      else {
        this.rapid_tasks = [];
      }
      if (initObj.hasOwnProperty('mechanical_units')) {
        this.mechanical_units = initObj.mechanical_units
      }
      else {
        this.mechanical_units = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SystemState
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [motors_on]
    bufferOffset = _serializer.bool(obj.motors_on, buffer, bufferOffset);
    // Serialize message field [auto_mode]
    bufferOffset = _serializer.bool(obj.auto_mode, buffer, bufferOffset);
    // Serialize message field [rapid_running]
    bufferOffset = _serializer.bool(obj.rapid_running, buffer, bufferOffset);
    // Serialize message field [rapid_tasks]
    // Serialize the length for message field [rapid_tasks]
    bufferOffset = _serializer.uint32(obj.rapid_tasks.length, buffer, bufferOffset);
    obj.rapid_tasks.forEach((val) => {
      bufferOffset = RAPIDTaskState.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [mechanical_units]
    // Serialize the length for message field [mechanical_units]
    bufferOffset = _serializer.uint32(obj.mechanical_units.length, buffer, bufferOffset);
    obj.mechanical_units.forEach((val) => {
      bufferOffset = MechanicalUnitState.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SystemState
    let len;
    let data = new SystemState(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [motors_on]
    data.motors_on = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [auto_mode]
    data.auto_mode = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [rapid_running]
    data.rapid_running = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [rapid_tasks]
    // Deserialize array length for message field [rapid_tasks]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.rapid_tasks = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.rapid_tasks[i] = RAPIDTaskState.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [mechanical_units]
    // Deserialize array length for message field [mechanical_units]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.mechanical_units = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.mechanical_units[i] = MechanicalUnitState.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.rapid_tasks.forEach((val) => {
      length += RAPIDTaskState.getMessageSize(val);
    });
    object.mechanical_units.forEach((val) => {
      length += MechanicalUnitState.getMessageSize(val);
    });
    return length + 11;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_robot_msgs/SystemState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '46309669ed6ba4863375b03174d00ad7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent a subset of states
    #   of an ABB robot controller system.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Metadata.
    std_msgs/Header header
    
    # Motors on/off indicator.
    bool motors_on
    
    # Auto/manual mode indicator.
    bool auto_mode
    
    # RAPID running/stopped indicator.
    bool rapid_running
    
    # RAPID task states.
    RAPIDTaskState[] rapid_tasks
    
    # Mechanical unit states.
    MechanicalUnitState[] mechanical_units
    
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
    MSG: abb_robot_msgs/RAPIDTaskState
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
    
    ================================================================================
    MSG: abb_robot_msgs/MechanicalUnitState
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent a subset of states
    #   of a mechanical unit defined in an ABB robot controller system.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The mechanical unit's name.
    string name
    
    # The mechanical unit's activated/deactivated status.
    bool activated
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SystemState(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.motors_on !== undefined) {
      resolved.motors_on = msg.motors_on;
    }
    else {
      resolved.motors_on = false
    }

    if (msg.auto_mode !== undefined) {
      resolved.auto_mode = msg.auto_mode;
    }
    else {
      resolved.auto_mode = false
    }

    if (msg.rapid_running !== undefined) {
      resolved.rapid_running = msg.rapid_running;
    }
    else {
      resolved.rapid_running = false
    }

    if (msg.rapid_tasks !== undefined) {
      resolved.rapid_tasks = new Array(msg.rapid_tasks.length);
      for (let i = 0; i < resolved.rapid_tasks.length; ++i) {
        resolved.rapid_tasks[i] = RAPIDTaskState.Resolve(msg.rapid_tasks[i]);
      }
    }
    else {
      resolved.rapid_tasks = []
    }

    if (msg.mechanical_units !== undefined) {
      resolved.mechanical_units = new Array(msg.mechanical_units.length);
      for (let i = 0; i < resolved.mechanical_units.length; ++i) {
        resolved.mechanical_units[i] = MechanicalUnitState.Resolve(msg.mechanical_units[i]);
      }
    }
    else {
      resolved.mechanical_units = []
    }

    return resolved;
    }
};

module.exports = SystemState;
