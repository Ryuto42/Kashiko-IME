cmake_minimum_required(VERSION 3.16)

project(fcitx5-kashiko VERSION 0.1.0)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(ECM REQUIRED)
set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Fcitx5Core 5.0 REQUIRED)
find_package(Fcitx5Module COMPONENTS Notifications QuickPhrase)
find_package(PkgConfig REQUIRED)
pkg_check_modules(XKBCommon REQUIRED xkbcommon)

include(FeatureSummary)
include(GNUInstallDirs)
include(ECMSetupVersion)
include(ECMUninstallTarget)

# Use fcitx5 compiler settings
include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake")

# Path to the Rust library
set(KASHIKO_RUST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release/libkashiko_fcitx5.so")
set(KASHIKO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Build Rust library as a custom target
find_program(CARGO cargo REQUIRED)
add_custom_target(kashiko_rust_lib ALL
    COMMAND ${CARGO} build --release -p kashiko-fcitx5
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    COMMENT "Building kashiko-fcitx5 Rust library"
    BYPRODUCTS "${KASHIKO_RUST_LIB}"
)

# Add the addon
add_library(kashiko MODULE
    src/kashiko.cpp
)
add_dependencies(kashiko kashiko_rust_lib)

target_include_directories(kashiko PRIVATE
    ${KASHIKO_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${XKBCommon_INCLUDE_DIRS}
)

# Expose fcitx5 version to C++ so we can guard newer API usage
if(Fcitx5Core_VERSION VERSION_GREATER_EQUAL "5.1.9")
    target_compile_definitions(kashiko PRIVATE FCITX5_HAS_CANDIDATE_SET_COMMENT)
endif()

target_link_libraries(kashiko
    Fcitx5::Core
    Fcitx5::Config
    ${KASHIKO_RUST_LIB}
    ${XKBCommon_LIBRARIES}
)

# Remove lib prefix (fcitx5 expects "kashiko.so" not "libkashiko.so")
set_target_properties(kashiko PROPERTIES PREFIX "")

# Set RPATH so fcitx5 can find libkashiko_fcitx5.so at runtime
# $ORIGIN means "same directory as this library"
set_target_properties(kashiko PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
    INSTALL_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH FALSE
)

# Install the addon
install(TARGETS kashiko DESTINATION "${FCITX_INSTALL_ADDONDIR}")

# Install Rust library alongside the addon
install(FILES "${KASHIKO_RUST_LIB}" DESTINATION "${FCITX_INSTALL_ADDONDIR}")

# Install addon config
install(FILES kashiko-addon.conf.in DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
    RENAME kashiko.conf)

# Install input method config
install(FILES kashiko.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod")

# Install metainfo
install(FILES org.fcitx.Fcitx5.Addon.Kashiko.metainfo.xml.in
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/metainfo"
    RENAME org.fcitx.Fcitx5.Addon.Kashiko.metainfo.xml)

# Install icons at multiple sizes for XDG icon theme
install(FILES icons/fcitx-kashiko-16.png
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/16x16/apps/"
    RENAME fcitx-kashiko.png)
install(FILES icons/fcitx-kashiko-24.png
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/24x24/apps/"
    RENAME fcitx-kashiko.png)
install(FILES icons/fcitx-kashiko-32.png
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/32x32/apps/"
    RENAME fcitx-kashiko.png)
install(FILES icons/fcitx-kashiko-48.png
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/48x48/apps/"
    RENAME fcitx-kashiko.png)
install(FILES icons/fcitx-kashiko-128.png
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/128x128/apps/"
    RENAME fcitx-kashiko.png)

feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)
