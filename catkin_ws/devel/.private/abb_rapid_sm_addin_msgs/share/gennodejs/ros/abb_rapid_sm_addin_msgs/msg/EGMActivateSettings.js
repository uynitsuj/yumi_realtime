// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let abb_rapid_msgs = _finder('abb_rapid_msgs');

//-----------------------------------------------------------

class EGMActivateSettings {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.tool = null;
      this.wobj = null;
      this.correction_frame = null;
      this.sensor_frame = null;
      this.cond_min_max = null;
      this.lp_filter = null;
      this.sample_rate = null;
      this.max_speed_deviation = null;
    }
    else {
      if (initObj.hasOwnProperty('tool')) {
        this.tool = initObj.tool
      }
      else {
        this.tool = new abb_rapid_msgs.msg.ToolData();
      }
      if (initObj.hasOwnProperty('wobj')) {
        this.wobj = initObj.wobj
      }
      else {
        this.wobj = new abb_rapid_msgs.msg.WObjData();
      }
      if (initObj.hasOwnProperty('correction_frame')) {
        this.correction_frame = initObj.correction_frame
      }
      else {
        this.correction_frame = new abb_rapid_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('sensor_frame')) {
        this.sensor_frame = initObj.sensor_frame
      }
      else {
        this.sensor_frame = new abb_rapid_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('cond_min_max')) {
        this.cond_min_max = initObj.cond_min_max
      }
      else {
        this.cond_min_max = 0.0;
      }
      if (initObj.hasOwnProperty('lp_filter')) {
        this.lp_filter = initObj.lp_filter
      }
      else {
        this.lp_filter = 0.0;
      }
      if (initObj.hasOwnProperty('sample_rate')) {
        this.sample_rate = initObj.sample_rate
      }
      else {
        this.sample_rate = 0;
      }
      if (initObj.hasOwnProperty('max_speed_deviation')) {
        this.max_speed_deviation = initObj.max_speed_deviation
      }
      else {
        this.max_speed_deviation = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EGMActivateSettings
    // Serialize message field [tool]
    bufferOffset = abb_rapid_msgs.msg.ToolData.serialize(obj.tool, buffer, bufferOffset);
    // Serialize message field [wobj]
    bufferOffset = abb_rapid_msgs.msg.WObjData.serialize(obj.wobj, buffer, bufferOffset);
    // Serialize message field [correction_frame]
    bufferOffset = abb_rapid_msgs.msg.Pose.serialize(obj.correction_frame, buffer, bufferOffset);
    // Serialize message field [sensor_frame]
    bufferOffset = abb_rapid_msgs.msg.Pose.serialize(obj.sensor_frame, buffer, bufferOffset);
    // Serialize message field [cond_min_max]
    bufferOffset = _serializer.float32(obj.cond_min_max, buffer, bufferOffset);
    // Serialize message field [lp_filter]
    bufferOffset = _serializer.float32(obj.lp_filter, buffer, bufferOffset);
    // Serialize message field [sample_rate]
    bufferOffset = _serializer.uint8(obj.sample_rate, buffer, bufferOffset);
    // Serialize message field [max_speed_deviation]
    bufferOffset = _serializer.float32(obj.max_speed_deviation, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EGMActivateSettings
    let len;
    let data = new EGMActivateSettings(null);
    // Deserialize message field [tool]
    data.tool = abb_rapid_msgs.msg.ToolData.deserialize(buffer, bufferOffset);
    // Deserialize message field [wobj]
    data.wobj = abb_rapid_msgs.msg.WObjData.deserialize(buffer, bufferOffset);
    // Deserialize message field [correction_frame]
    data.correction_frame = abb_rapid_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [sensor_frame]
    data.sensor_frame = abb_rapid_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [cond_min_max]
    data.cond_min_max = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [lp_filter]
    data.lp_filter = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [sample_rate]
    data.sample_rate = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [max_speed_deviation]
    data.max_speed_deviation = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += abb_rapid_msgs.msg.WObjData.getMessageSize(object.wobj);
    return length + 142;
  }

  static datatype() {
    // Returns string type for a message object
    return 'abb_rapid_sm_addin_msgs/EGMActivateSettings';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'edc93be8ce076e9147c54b4dc80e7974';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMActivateSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMActivateSettings" are for example used with the RAPID EGM instructions
    #   "EGMActJoint" and "EGMActPose".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Tool to use for EGM motions.
    abb_rapid_msgs/ToolData tool
    
    # Work object to use for EGM motions.
    abb_rapid_msgs/WObjData wobj
    
    # Correction frame to use for EGM motions (only used in EGM pose mode).
    abb_rapid_msgs/Pose correction_frame
    
    # Sensor frame to use for EGM motions (only used in EGM pose mode).
    abb_rapid_msgs/Pose sensor_frame
    
    # Condition threshold value [deg or mm] for when an EGM motion's convergence
    # criteria is considered to be fulfilled.
    float32 cond_min_max
    
    # Low pass filter bandwidth [Hz] for the EGM controller.
    float32 lp_filter
    
    # Sample rate [ms] for the EGM communication.
    uint8 sample_rate
    
    # Maximum admitted joint speed [deg/s].
    float32 max_speed_deviation
    
    ================================================================================
    MSG: abb_rapid_msgs/ToolData
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
    
    ================================================================================
    MSG: abb_rapid_msgs/WObjData
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EGMActivateSettings(null);
    if (msg.tool !== undefined) {
      resolved.tool = abb_rapid_msgs.msg.ToolData.Resolve(msg.tool)
    }
    else {
      resolved.tool = new abb_rapid_msgs.msg.ToolData()
    }

    if (msg.wobj !== undefined) {
      resolved.wobj = abb_rapid_msgs.msg.WObjData.Resolve(msg.wobj)
    }
    else {
      resolved.wobj = new abb_rapid_msgs.msg.WObjData()
    }

    if (msg.correction_frame !== undefined) {
      resolved.correction_frame = abb_rapid_msgs.msg.Pose.Resolve(msg.correction_frame)
    }
    else {
      resolved.correction_frame = new abb_rapid_msgs.msg.Pose()
    }

    if (msg.sensor_frame !== undefined) {
      resolved.sensor_frame = abb_rapid_msgs.msg.Pose.Resolve(msg.sensor_frame)
    }
    else {
      resolved.sensor_frame = new abb_rapid_msgs.msg.Pose()
    }

    if (msg.cond_min_max !== undefined) {
      resolved.cond_min_max = msg.cond_min_max;
    }
    else {
      resolved.cond_min_max = 0.0
    }

    if (msg.lp_filter !== undefined) {
      resolved.lp_filter = msg.lp_filter;
    }
    else {
      resolved.lp_filter = 0.0
    }

    if (msg.sample_rate !== undefined) {
      resolved.sample_rate = msg.sample_rate;
    }
    else {
      resolved.sample_rate = 0
    }

    if (msg.max_speed_deviation !== undefined) {
      resolved.max_speed_deviation = msg.max_speed_deviation;
    }
    else {
      resolved.max_speed_deviation = 0.0
    }

    return resolved;
    }
};

module.exports = EGMActivateSettings;
