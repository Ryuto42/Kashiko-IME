//! kashiko-im: Japanese IME engine for the macOS frontend.
//!
//! The stdio JSON-RPC server lives in [`server`] and is built as the
//! `kashiko-imserver` binary, bundled inside `kashiko-macos`.

pub mod config;
pub mod core;
pub mod server;

pub use core::engine::{
    DeepSearchJob, DeepSearchOutcome, EngineAction, EngineResult, InputMethodEngine, RefineJob,
    RefineOutcome, run_deep_job, run_refine_job,
};
pub use core::keycode::{KeyEvent, KeyModifiers, Keysym};
pub use core::state::InputState;
