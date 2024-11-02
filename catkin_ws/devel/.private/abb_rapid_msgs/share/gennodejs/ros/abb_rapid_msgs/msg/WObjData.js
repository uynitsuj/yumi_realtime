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

//-----------------------------------------------------------

class WObjData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.robhold = null;
      this.ufprog = null;
      this.ufmec = null;
      this.uframe = null;
      this.oframe = null;
    }
    else {
      if (initObj.hasOwnProperty('robhold')) {
        this.robhold = initObj.robhold
      }
      else {
        this.robhold = false;
      }
      if (initObj.hasOwnProperty('ufprog')) {
        this.ufprog = initObj.ufprog
      }
      else {
        this.ufprog = false;
      }
      if (initObj.hasOwnProperty('ufmec')) {
        this.ufmec = initObj.ufmec
      }
      else {
        this.ufmec = '';
      }
      if (initObj.hasOwnProperty('uframe')) {
        this.uframe = initObj.uframe
      }
      else {
        this.uframe = new Pose();
      }
      if (initObj.hasOwnProperty('oframe')) {
        this.oframe = initObj.oframe
      }
      else {
        this.oframe = new Pose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type WObjData
    // Serialize message field [robhold]
    bufferOffset = _serializer.bool(obj.robhold, buffer, bufferOffset);
    // Serialize message field [ufprog]
    bufferOffset = _serializer.bool(obj.ufprog, buffer, bufferOffset);
    // Serialize message field [ufmec]
    bufferOffset = _serializer.string(obj.ufmec, buffer, bufferOffset);
    // Serialize message field [uframe]
    bufferOffset = Pose.serialize(obj.uframe, buffer, bufferOffset);
    // Serialize message field [oframe]
    bufferOffset = Pose.serialize(obj.oframe, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type WObjData
    let len;
    let data = new WObjData(null);
    // Deserialize message field [robhold]
    data.robhold = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [ufprog]
    data.ufprog = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [ufmec]
    data.ufmec = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [uframe]
    data.uframe = Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [oframe]
    data.oframe = Pose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.ufmec);
    return length + 62;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/WObjData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eb6fd21e8b900ca32b7fe6c07697ac07';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "wobjdata" (work object data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Defines whether or not the robot in the actual
    # program task is holding the work object:
    # - TRUE: The robot is holding the work object,
    #         i.e. using a stationary tool.
    # - FALSE: The robot is not holding the work object,
    #          i.e. the robot is holding the tool.
    bool robhold
    
    # Defines whether or not a fixed user coordinate system is used:
    # - TRUE: Fixed user coordinate system.
    # - FALSE: Movable user coordinate system,
    #          i.e. coordinated external axes are used.
    #          Also to be used in a MultiMove system in
    #          semicoordinated or synchronized coordinated mode.
    bool ufprog
    
    # The mechanical unit with which the robot movements are coordinated.
    #
    # Only specified in the case of movable user
    # coordinate systems (ufprog is FALSE).
    string ufmec
    
    # The user coordinate system, i.e. the position
    # of the current work surface or fixture.
    Pose uframe
    
    # The object coordinate system, i.e. the position
    # of the current work object.
    Pose oframe
    
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new WObjData(null);
    if (msg.robhold !== undefined) {
      resolved.robhold = msg.robhold;
    }
    else {
      resolved.robhold = false
    }

    if (msg.ufprog !== undefined) {
      resolved.ufprog = msg.ufprog;
    }
    else {
      resolved.ufprog = false
    }

    if (msg.ufmec !== undefined) {
      resolved.ufmec = msg.ufmec;
    }
    else {
      resolved.ufmec = ''
    }

    if (msg.uframe !== undefined) {
      resolved.uframe = Pose.Resolve(msg.uframe)
    }
    else {
      resolved.uframe = new Pose()
    }

    if (msg.oframe !== undefined) {
      resolved.oframe = Pose.Resolve(msg.oframe)
    }
    else {
      resolved.oframe = new Pose()
    }

    return resolved;
    }
};

module.exports = WObjData;
