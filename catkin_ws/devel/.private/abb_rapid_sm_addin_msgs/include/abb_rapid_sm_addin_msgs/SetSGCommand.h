// Generated by gencpp from file abb_rapid_sm_addin_msgs/SetSGCommand.msg
// DO NOT EDIT!


#ifndef ABB_RAPID_SM_ADDIN_MSGS_MESSAGE_SETSGCOMMAND_H
#define ABB_RAPID_SM_ADDIN_MSGS_MESSAGE_SETSGCOMMAND_H

#include <ros/service_traits.h>


#include <abb_rapid_sm_addin_msgs/SetSGCommandRequest.h>
#include <abb_rapid_sm_addin_msgs/SetSGCommandResponse.h>


namespace abb_rapid_sm_addin_msgs
{

struct SetSGCommand
{

typedef SetSGCommandRequest Request;
typedef SetSGCommandResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SetSGCommand
} // namespace abb_rapid_sm_addin_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommand > {
  static const char* value()
  {
    return "7aa352af5c8c7b889375c50673d12253";
  }

  static const char* value(const ::abb_rapid_sm_addin_msgs::SetSGCommand&) { return value(); }
};

template<>
struct DataType< ::abb_rapid_sm_addin_msgs::SetSGCommand > {
  static const char* value()
  {
    return "abb_rapid_sm_addin_msgs/SetSGCommand";
  }

  static const char* value(const ::abb_rapid_sm_addin_msgs::SetSGCommand&) { return value(); }
};


// service_traits::MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommandRequest> should match
// service_traits::MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommand >
template<>
struct MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommandRequest>
{
  static const char* value()
  {
    return MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommand >::value();
  }
  static const char* value(const ::abb_rapid_sm_addin_msgs::SetSGCommandRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::abb_rapid_sm_addin_msgs::SetSGCommandRequest> should match
// service_traits::DataType< ::abb_rapid_sm_addin_msgs::SetSGCommand >
template<>
struct DataType< ::abb_rapid_sm_addin_msgs::SetSGCommandRequest>
{
  static const char* value()
  {
    return DataType< ::abb_rapid_sm_addin_msgs::SetSGCommand >::value();
  }
  static const char* value(const ::abb_rapid_sm_addin_msgs::SetSGCommandRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommandResponse> should match
// service_traits::MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommand >
template<>
struct MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommandResponse>
{
  static const char* value()
  {
    return MD5Sum< ::abb_rapid_sm_addin_msgs::SetSGCommand >::value();
  }
  static const char* value(const ::abb_rapid_sm_addin_msgs::SetSGCommandResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::abb_rapid_sm_addin_msgs::SetSGCommandResponse> should match
// service_traits::DataType< ::abb_rapid_sm_addin_msgs::SetSGCommand >
template<>
struct DataType< ::abb_rapid_sm_addin_msgs::SetSGCommandResponse>
{
  static const char* value()
  {
    return DataType< ::abb_rapid_sm_addin_msgs::SetSGCommand >::value();
  }
  static const char* value(const ::abb_rapid_sm_addin_msgs::SetSGCommandResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // ABB_RAPID_SM_ADDIN_MSGS_MESSAGE_SETSGCOMMAND_H