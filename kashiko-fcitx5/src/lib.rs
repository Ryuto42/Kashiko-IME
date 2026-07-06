//! kashiko-fcitx5: fcitx5 addon for the Kashiko Japanese IME
//!
//! This crate provides the C FFI interface consumed by the fcitx5 C++ addon
//! (`fcitx5-addon/src/kashiko.cpp`). It wraps kashiko-im's engine and exposes
//! a stable C ABI for input handling, preedit, candidates, and learning.

pub mod ffi;
