// Auto-generated. Do not edit!

// (in-package abb_rapid_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let Pose = require('./Pose.js');
let LoadData = require('./LoadData.js');

//-----------------------------------------------------------

class ToolData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.robhold = null;
      this.tframe = null;
      this.tload = null;
    }
    else {
      if (initObj.hasOwnProperty('robhold')) {
        this.robhold = initObj.robhold
      }
      else {
        this.robhold = false;
      }
      if (initObj.hasOwnProperty('tframe')) {
        this.tframe = initObj.tframe
      }
      else {
        this.tframe = new Pose();
      }
      if (initObj.hasOwnProperty('tload')) {
        this.tload = initObj.tload
      }
      else {
        this.tload = new LoadData();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ToolData
    // Serialize message field [robhold]
    bufferOffset = _serializer.bool(obj.robhold, buffer, bufferOffset);
    // Serialize message field [tframe]
    bufferOffset = Pose.serialize(obj.tframe, buffer, bufferOffset);
    // Serialize message field [tload]
    bufferOffset = LoadData.serialize(obj.tload, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ToolData
    let len;
    let data = new ToolData(null);
    // Deserialize message field [robhold]
    data.robhold = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [tframe]
    data.tframe = Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [tload]
    data.tload = LoadData.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 73;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/ToolData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2d526e4dd17e28147b27f746cf88c5b6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "tooldata" (tool data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Defines whether or not the robot is holding the tool:
    # - TRUE: The robot is holding the tool.
    # - FALSE: The robot is not holding the tool, that is, a stationary tool.
    bool robhold
    
    # The tool coordinate system, that is:
    # - The position of the TCP (x, y and z) in mm,
    #   expressed in the wrist coordinate system (tool0).
    # - The orientation of the tool coordinate system,
    #   expressed in the wrist coordinate system.
    Pose tframe
    
    # The load of the tool (if robot held tool).
    # The load of the gripper holding the work object (if stationary tool).
    LoadData tload
    
    ================================================================================
    MSG: abb_rapid_msgs/Pose
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "pose" (coordinate transformations).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The displacement in position (x, y, and z) of the coordinate system.
    Pos trans
    
    # The rotation of the coordinate system.
    Orient rot
    
    ================================================================================
    MSG: abb_rapid_msgs/Pos
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "pos" (positions (only X, Y and Z)).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The x-, y- and z-value of the position.
    float32 x
    float32 y
    float32 z
    
    ================================================================================
    MSG: abb_rapid_msgs/Orient
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "orient" (orientation).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Quaternion 1-4.
    float32 q1
    float32 q2
    float32 q3
    float32 q4
    
    ================================================================================
    MSG: abb_rapid_msgs/LoadData
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "loaddata" (load data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The mass (weight) of the load in kg.
    float32 mass
    
    # The center of gravity of the payload, expressed in mm,
    # in the tool coordinate system if the robot is holding the tool.
    #
    # If a stationary tool is used then the center of gravity for the payload
    # held by the gripper is expressed in the object frame of the work object
    # coordinate system moved by the robot.
    Pos cog
    
    # The orientation of the axes of moment.
    #
    # These are the principal axes of the payload
    # moment of inertia with origin in cog.
    #
    # If the robot is holding the tool, the axes of moment
    # are expressed in the tool coordinate system.
    Orient aom
    
    # The moment of inertia of the load around the x-, y- and z-axis,
    # expressed in kgm^2.
    float32 ix
    float32 iy
    float32 iz
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ToolData(null);
    if (msg.robhold !== undefined) {
      resolved.robhold = msg.robhold;
    }
    else {
      resolved.robhold = false
    }

    if (msg.tframe !== undefined) {
      resolved.tframe = Pose.Resolve(msg.tframe)
    }
    else {
      resolved.tframe = new Pose()
    }

    if (msg.tload !== undefined) {
      resolved.tload = LoadData.Resolve(msg.tload)
    }
    else {
      resolved.tload = new LoadData()
    }

    return resolved;
    }
};

module.exports = ToolData;
