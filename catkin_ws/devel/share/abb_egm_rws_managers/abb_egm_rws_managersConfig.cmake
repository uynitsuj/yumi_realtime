########################################################################################################################
## Configuration file for 'abb_egm_rws_managers'
########################################################################################################################
include(CMakeFindDependencyMacro)

#===========================================================
# Dependencies
#===========================================================
find_dependency(abb_libegm REQUIRED)
find_dependency(abb_librws REQUIRED)
find_dependency(Protobuf REQUIRED)
find_dependency(Threads REQUIRED)

# Our library dependencies (contains definitions for IMPORTED targets).
include("${CMAKE_CURRENT_LIST_DIR}/abb_egm_rws_managersTargets.cmake")

# Set legacy _LIBRARIES variable (e.g. due to that 'ament_cmake'
# seems to need it), however it should not be required with
# modern CMake (i.e. targets and properties).
set(abb_egm_rws_managers_LIBRARIES abb_egm_rws_managers::abb_egm_rws_managers)
