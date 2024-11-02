// Auto-generated. Do not edit!

// (in-package abb_rapid_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let Pos = require('./Pos.js');
let Orient = require('./Orient.js');

//-----------------------------------------------------------

class LoadData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.mass = null;
      this.cog = null;
      this.aom = null;
      this.ix = null;
      this.iy = null;
      this.iz = null;
    }
    else {
      if (initObj.hasOwnProperty('mass')) {
        this.mass = initObj.mass
      }
      else {
        this.mass = 0.0;
      }
      if (initObj.hasOwnProperty('cog')) {
        this.cog = initObj.cog
      }
      else {
        this.cog = new Pos();
      }
      if (initObj.hasOwnProperty('aom')) {
        this.aom = initObj.aom
      }
      else {
        this.aom = new Orient();
      }
      if (initObj.hasOwnProperty('ix')) {
        this.ix = initObj.ix
      }
      else {
        this.ix = 0.0;
      }
      if (initObj.hasOwnProperty('iy')) {
        this.iy = initObj.iy
      }
      else {
        this.iy = 0.0;
      }
      if (initObj.hasOwnProperty('iz')) {
        this.iz = initObj.iz
      }
      else {
        this.iz = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type LoadData
    // Serialize message field [mass]
    bufferOffset = _serializer.float32(obj.mass, buffer, bufferOffset);
    // Serialize message field [cog]
    bufferOffset = Pos.serialize(obj.cog, buffer, bufferOffset);
    // Serialize message field [aom]
    bufferOffset = Orient.serialize(obj.aom, buffer, bufferOffset);
    // Serialize message field [ix]
    bufferOffset = _serializer.float32(obj.ix, buffer, bufferOffset);
    // Serialize message field [iy]
    bufferOffset = _serializer.float32(obj.iy, buffer, bufferOffset);
    // Serialize message field [iz]
    bufferOffset = _serializer.float32(obj.iz, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type LoadData
    let len;
    let data = new LoadData(null);
    // Deserialize message field [mass]
    data.mass = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cog]
    data.cog = Pos.deserialize(buffer, bufferOffset);
    // Deserialize message field [aom]
    data.aom = Orient.deserialize(buffer, bufferOffset);
    // Deserialize message field [ix]
    data.ix = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [iy]
    data.iy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [iz]
    data.iz = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 44;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/LoadData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'dfe575484752eff5e837f54d2fd939e4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new LoadData(null);
    if (msg.mass !== undefined) {
      resolved.mass = msg.mass;
    }
    else {
      resolved.mass = 0.0
    }

    if (msg.cog !== undefined) {
      resolved.cog = Pos.Resolve(msg.cog)
    }
    else {
      resolved.cog = new Pos()
    }

    if (msg.aom !== undefined) {
      resolved.aom = Orient.Resolve(msg.aom)
    }
    else {
      resolved.aom = new Orient()
    }

    if (msg.ix !== undefined) {
      resolved.ix = msg.ix;
    }
    else {
      resolved.ix = 0.0
    }

    if (msg.iy !== undefined) {
      resolved.iy = msg.iy;
    }
    else {
      resolved.iy = 0.0
    }

    if (msg.iz !== undefined) {
      resolved.iz = msg.iz;
    }
    else {
      resolved.iz = 0.0
    }

    return resolved;
    }
};

module.exports = LoadData;
