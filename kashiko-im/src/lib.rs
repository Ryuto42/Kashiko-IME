//! kashiko-im: Japanese IME engine shared by the fcitx5 (Linux) and
//! macOS frontends.
//!
//! - fcitx5 C FFI lives in the separate `kashiko-fcitx5` crate.
//! - The macOS stdio JSON-RPC server lives in [`server`] and is built
//!   as the `kashiko-imserver` binary, bundled inside `kashiko-macos`.

pub mod config;
pub mod core;
pub mod server;

pub use core::engine::{
    DeepSearchJob, DeepSearchOutcome, EngineAction, EngineResult, InputMethodEngine, run_deep_job,
};
pub use core::keycode::{KeyEvent, KeyModifiers, Keysym};
pub use core::state::InputState;
