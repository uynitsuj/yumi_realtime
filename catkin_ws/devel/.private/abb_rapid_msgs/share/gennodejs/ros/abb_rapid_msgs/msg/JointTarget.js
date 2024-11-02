// Auto-generated. Do not edit!

// (in-package abb_rapid_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let RobJoint = require('./RobJoint.js');
let ExtJoint = require('./ExtJoint.js');

//-----------------------------------------------------------

class JointTarget {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.robax = null;
      this.extax = null;
    }
    else {
      if (initObj.hasOwnProperty('robax')) {
        this.robax = initObj.robax
      }
      else {
        this.robax = new RobJoint();
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
    // Serializes a message object of type JointTarget
    // Serialize message field [robax]
    bufferOffset = RobJoint.serialize(obj.robax, buffer, bufferOffset);
    // Serialize message field [extax]
    bufferOffset = ExtJoint.serialize(obj.extax, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type JointTarget
    let len;
    let data = new JointTarget(null);
    // Deserialize message field [robax]
    data.robax = RobJoint.deserialize(buffer, bufferOffset);
    // Deserialize message field [extax]
    data.extax = ExtJoint.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 48;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_msgs/JointTarget';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a80f8974c2ebed9e2ef462fcc2a28a38';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "jointtarget" (joint position data).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Axis positions of the robot axes in degrees.
    RobJoint robax
    
    # The position of the external axes.
    ExtJoint extax
    
    ================================================================================
    MSG: abb_rapid_msgs/RobJoint
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the
    #   RAPID data type "robjoint" (joint position of robot axes).
    #
    # Note:
    #   Please see the "Technical reference manual - RAPID Instructions,
    #   Functions and Data types" for more details.
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # The position of robot axis 1-6 in degrees from the calibration position.
    float32 rax_1
    float32 rax_2
    float32 rax_3
    float32 rax_4
    float32 rax_5
    float32 rax_6
    
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
    const resolved = new JointTarget(null);
    if (msg.robax !== undefined) {
      resolved.robax = RobJoint.Resolve(msg.robax)
    }
    else {
      resolved.robax = new RobJoint()
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

module.exports = JointTarget;
