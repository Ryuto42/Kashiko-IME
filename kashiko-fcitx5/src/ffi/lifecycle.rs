#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_int;

use super::{KashikoEngine, ffi_mut, init_logging};

/// Create a new Kashiko engine instance
/// Returns a pointer to the engine, or null on failure
#[unsafe(no_mangle)]
pub extern "C" fn kashiko_engine_new() -> *mut KashikoEngine {
    init_logging();
    let engine = Box::new(KashikoEngine::new());
    Box::into_raw(engine)
}

/// Initialize the kanji converter (loads the model)
/// Returns 0 on success, -1 on failure
#[unsafe(no_mangle)]
pub extern "C" fn kashiko_engine_init(engine: *mut KashikoEngine) -> c_int {
    let engine = ffi_mut!(engine, -1);
    match engine.engine.init_from_settings(&engine.settings) {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("Kashiko init failed: {:#}", e);
            -1
        }
    }
}

/// Destroy a Kashiko engine instance
#[unsafe(no_mangle)]
pub extern "C" fn kashiko_engine_free(engine: *mut KashikoEngine) {
    if !engine.is_null() {
        let engine_ref = unsafe { &mut *engine };
        engine_ref.engine.save_learning();
        // SAFETY: Pointer is non-null (checked above) and was created by Box::into_raw in kashiko_engine_new
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}
