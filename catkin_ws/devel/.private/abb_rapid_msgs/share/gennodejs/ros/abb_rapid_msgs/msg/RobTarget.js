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
let ConfData = require('./ConfData.js');
let ExtJoint = require('./ExtJoint.js');

//-----------------------------------------------------------

class RobTarget {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.trans = null;
      this.rot = null;
      this.robconf = null;
      this.extax = null;
    }
    else {
      if (initObj.hasOwnProperty('trans')) {
        this.trans = initObj.trans
      }
      else {
        this.trans = new Pos();
      }
      if (initObj.hasOwnProperty('rot')) {
        this.rot = initObj.rot
      }
      else {
        this.rot = new Orient();
      }
      if (initObj.hasOwnProperty('robconf')) {
        this.robconf = initObj.robconf
      }
      else {
        this.robconf = new ConfData();
      }
      if (initObj.hasOwnProperty('extax')) {
        this.extax = initObj.extax
      }
      else {
        this.extax = new ExtJoint();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RobTarget
    // Serialize message field [trans]
    bufferOffset = Pos.serialize(obj.trans, buffer, bufferOffset);
    // Serialize message field [rot]
    bufferOffset = Orient.serialize(obj.rot, buffer, bufferOffset);
    // Serialize message field [robconf]
    bufferOffset = ConfData.serialize(obj.robconf, buffer, bufferOffset);
    // Serialize message field [extax]
    bufferOffset = ExtJoint.serialize(obj.extax, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RobTarget
    let len;
    let data = new RobTarget(null);
    // Deserialize message field [trans]
    data.trans = Pos.deserialize(buffer, bufferOffset);
    // Deserialize message field [rot]
    data.rot = Orient.deserialize(buffer, bufferOffset);
    // Deserialize message field [robconf]
    data.robconf = ConfData.deserialize(buffer, bufferOffset);
    // Deserialize message field [extax]
    data.extax = ExtJoint.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 68;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/RobTarget';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '31d5c046c0fae9eadd8133382f15db0a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "robtarget" (position data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The position (x, y, and z) of the tool center point expressed in mm.
    #
    # The position is specified in relation to the current object coordinate system,
    # including program displacement.
    #
    # If no work object is specified then this is the world coordinate system.
    Pos trans
    
    # The orientation of the tool, expressed in the
    # form of a quaternion (q1, q2, q3, and q4).
    #
    # The orientation is specified in relation to the current
    # object coordinate system including program displacement.
    #
    # If no work object is specified then this is the world coordinate system.
    Orient rot
    
    # The axis configuration of the robot (cf1, cf4, cf6, and cfx).
    #
    # This is defined in the form of the current quarter
    # revolution of axis 1, axis 4, and axis 6.
    #
    # The first positive quarter revolution 0 to 90 degrees is defined as 0.
    #
    # The meaning of the component cfx is dependent on robot type.
    ConfData robconf
    
    # The position of the additional axes.
    ExtJoint extax
    
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
    MSG: abb_rapid_msgs/ConfData
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "confdata" (robot configuration data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Rotating axis:
    # - The current quadrant of axis 1, 4 and 6,
    #   expressed as a positive or negative integer.
    # Linear axis:
    # - The current meter interval of axis 1, 4 and 6,
    #   expressed as a positive or negative integer.
    float32 cf1
    float32 cf4
    float32 cf6
    
    # Rotating axis:
    # - For serial link robots, the current robot configuration,
    #   expressed as an integer in the range from 0 to 7.
    # - For SCARA robots, the current robot configuration,
    #   expressed as an integer in the range from 0 to 1.
    # - For 7-axis robots, the current robot configuration,
    #   expressed as an integer in the range from 0 to 7.
    # - For paint robots, the current quadrant of axis 5,
    #   expressed as a positive or negative integer.
    # - For other robots, using the current quadrant of axis 2,
    #   expressed as a positive or negative integer.
    # Linear axis:
    # - The current meter interval of axis 2,
    #   expressed as a positive or negative integer.
    float32 cfx
    
    ================================================================================
    MSG: abb_rapid_msgs/ExtJoint
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "extjoint" (position of external joints).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The position of the external logical axis "a"-"f",
    # expressed in degrees or mm (depending on the type of axis).
    float32 eax_a
    float32 eax_b
    float32 eax_c
    float32 eax_d
    float32 eax_e
    float32 eax_f
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RobTarget(null);
    if (msg.trans !== undefined) {
      resolved.trans = Pos.Resolve(msg.trans)
    }
    else {
      resolved.trans = new Pos()
    }

    if (msg.rot !== undefined) {
      resolved.rot = Orient.Resolve(msg.rot)
    }
    else {
      resolved.rot = new Orient()
    }

    if (msg.robconf !== undefined) {
      resolved.robconf = ConfData.Resolve(msg.robconf)
    }
    else {
      resolved.robconf = new ConfData()
    }

    if (msg.extax !== undefined) {
      resolved.extax = ExtJoint.Resolve(msg.extax)
    }
    else {
      resolved.extax = new ExtJoint()
    }

    return resolved;
    }
};

module.exports = RobTarget;
