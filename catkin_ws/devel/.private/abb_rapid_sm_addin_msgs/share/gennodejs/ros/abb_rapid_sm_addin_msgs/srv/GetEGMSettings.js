// Auto-generated. Do not edit!

// (in-package abb_rapid_sm_addin_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let EGMSettings = require('../msg/EGMSettings.js');

//-----------------------------------------------------------

class GetEGMSettingsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.task = null;
    }
    else {
      if (initObj.hasOwnProperty('task')) {
        this.task = initObj.task
      }
      else {
        this.task = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetEGMSettingsRequest
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetEGMSettingsRequest
    let len;
    let data = new GetEGMSettingsRequest(null);
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.task);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_rapid_sm_addin_msgs/GetEGMSettingsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0ece8f504419f7ca4d91b277e47ff617';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this service definition, is to define a way to get the value
    #   of a predefined RAPID variable of the custom RAPID data type "EGMSettings".
    #
    # Note:
    #   This variable is defined in each RAPID motion task, in ABB robot controller
    #   systems, that are using the RobotWare "StateMachine Add-In".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Service request fields
    #-------------------------------------------------------------------------------
    # The RAPID task, where the variable exists.
    string task
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetEGMSettingsRequest(null);
    if (msg.task !== undefined) {
      resolved.task = msg.task;
    }
    else {
      resolved.task = ''
    }

    return resolved;
    }
};

class GetEGMSettingsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.settings = null;
      this.result_code = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('settings')) {
        this.settings = initObj.settings
      }
      else {
        this.settings = new EGMSettings();
      }
      if (initObj.hasOwnProperty('result_code')) {
        this.result_code = initObj.result_code
      }
      else {
        this.result_code = 0;
      }
      if (initObj.hasOwnProperty('message')) {
        this.message = initObj.message
      }
      else {
        this.message = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetEGMSettingsResponse
    // Serialize message field [settings]
    bufferOffset = EGMSettings.serialize(obj.settings, buffer, bufferOffset);
    // Serialize message field [result_code]
    bufferOffset = _serializer.uint16(obj.result_code, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetEGMSettingsResponse
    let len;
    let data = new GetEGMSettingsResponse(null);
    // Deserialize message field [settings]
    data.settings = EGMSettings.deserialize(buffer, bufferOffset);
    // Deserialize message field [result_code]
    data.result_code = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += EGMSettings.getMessageSize(object.settings);
    length += _getByteLength(object.message);
    return length + 6;
  }

  static datatype() {
    // Returns string type for a service object
    return 'abb_rapid_sm_addin_msgs/GetEGMSettingsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e1f4c6b9730a3324dc0ac94d49b23bf7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    #-------------------------------------------------------------------------------
    # Service response fields
    #-------------------------------------------------------------------------------
    # The settings used for RAPID EGM instructions.
    EGMSettings settings
    
    # Service success/failure indicator.
    # Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
    uint16 result_code
    
    # Status message (empty if service succeeded).
    string message
    
    
    ================================================================================
    MSG: abb_rapid_sm_addin_msgs/EGMSettings
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMSettings" from the RobotWare "StateMachine Add-In".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Indicator for if EGM motions are allowed to start.
    bool allow_egm_motions
    
    # Indicator for if RAPID tasks should be synced before starting EGM motions
    # (only used in multi robot systems).
    bool use_presync
    
    # Settings for EGMSetupUC instructions.
    EGMSetupSettings setup_uc
    
    # Settings for EGMAct instructions.
    EGMActivateSettings activate
    
    # Settings for EGMRun instructions.
    EGMRunSettings run
    
    # Settings for EGMStop instructions.
    EGMStopSettings stop
    
    ================================================================================
    MSG: abb_rapid_sm_addin_msgs/EGMSetupSettings
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMSetupSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMSetupSettings" are for example used with the RAPID EGM instruction
    #   "EGMSetupUC".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Indicator for if the EGM controller should apply filtering on the
    # EGM references.
    bool use_filtering
    
    # Communication timeout [s].
    float32 comm_timeout
    
    ================================================================================
    MSG: abb_rapid_sm_addin_msgs/EGMActivateSettings
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
    
    ================================================================================
    MSG: abb_rapid_sm_addin_msgs/EGMRunSettings
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMRunSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMRunSettings" are for example used with the RAPID EGM instructions
    #   "EGMRunJoint" and "EGMRunPose".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Condition time [s] for how long an EGM motion's convergence criteria must be
    # fulfilled before the motion is considered to be finished.
    float32 cond_time
    
    # Ramp in time [s].
    float32 ramp_in_time
    
    # Static offset applied on top of the EGM references
    # (only used in EGM pose mode).
    abb_rapid_msgs/Pose offset
    
    # Position correction gain for the EGM controller.
    float32 pos_corr_gain
    
    ================================================================================
    MSG: abb_rapid_sm_addin_msgs/EGMStopSettings
    #-------------------------------------------------------------------------------
    # Description:
    #   The purpose of this message definition, is to represent the custom RAPID
    #   data type "EGMStopSettings" from the RobotWare "StateMachine Add-In".
    #
    # Note:
    #   "EGMStopSettings" are for example used with the RAPID EGM instruction
    #   "EGMStop".
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    # Message fields
    #-------------------------------------------------------------------------------
    # Ramp out time [s].
    float32 ramp_out_time
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetEGMSettingsResponse(null);
    if (msg.settings !== undefined) {
      resolved.settings = EGMSettings.Resolve(msg.settings)
    }
    else {
      resolved.settings = new EGMSettings()
    }

    if (msg.result_code !== undefined) {
      resolved.result_code = msg.result_code;
    }
    else {
      resolved.result_code = 0
    }

    if (msg.message !== undefined) {
      resolved.message = msg.message;
    }
    else {
      resolved.message = ''
    }

    return resolved;
    }
};

module.exports = {
  Request: GetEGMSettingsRequest,
  Response: GetEGMSettingsResponse,
  md5sum() { return '8735578c5cf9e61a67a68c6c0fe0172b'; },
  datatype() { return 'abb_rapid_sm_addin_msgs/GetEGMSettings'; }
};
