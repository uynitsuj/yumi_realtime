# - Config file for the abb_libegm package
# It defines the following variable
#  abb_libegm_LIBRARIES - libraries to link against

include(CMakeFindDependencyMacro)

# Find dependencies
find_dependency(Threads REQUIRED)
find_dependency(Boost REQUIRED COMPONENTS regex system thread)

# Our library dependencies (contains definitions for IMPORTED targets)
include("${CMAKE_CURRENT_LIST_DIR}/abb_libegmTargets.cmake")

# These are IMPORTED targets created by abb_libegmTargets.cmake
set(abb_libegm_LIBRARIES abb_libegm::abb_libegm)
