//! Stdio JSON-RPC server wrapping [`InputMethodEngine`].
//!
//! This is the platform-independent core of `kashiko-imserver`, the engine
//! process spawned by the macOS (Swift/InputMethodKit) frontend. See
//! [`protocol`] for the wire format.

pub mod protocol;

use serde_json::{Value, json};

use crate::config::Settings;
use crate::core::candidate::CandidateList;
use crate::core::engine::{
    DeepSearchJob, DeepSearchOutcome, EngineAction, EngineConfig, EngineResult, InputMethodEngine,
};
use crate::core::keycode::{KeyEvent, Keysym};
use crate::core::state::InputState;

use protocol::{
    Action, CandidateItem, InitResult, KeyResult, PROTOCOL_VERSION, PreeditAttr, ProcessKeyParams,
    Request, Response, RpcError, SecureInputParams, SelectCandidateParams, StatusResult,
    SurroundingTextParams,
};

/// JSON-RPC dispatcher owning the engine instance.
///
/// One server serves one frontend over one stdin/stdout pair; engine state
/// (Empty → Composing → Conversion) lives here across requests.
pub struct ImServer {
    engine: InputMethodEngine,
    /// Pending settings, consumed by the first successful `init`.
    settings: Option<Settings>,
    initialized: bool,
}

impl Default for ImServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ImServer {
    /// Create a server with settings loaded from config.toml (or defaults).
    pub fn new() -> Self {
        let settings = Settings::load().unwrap_or_default();
        Self::with_settings(settings)
    }

    pub fn with_settings(settings: Settings) -> Self {
        let config = EngineConfig::from_settings(&settings);
        Self {
            engine: InputMethodEngine::with_config(config),
            settings: Some(settings),
            initialized: false,
        }
    }

    /// Save the learning cache (called on EOF/shutdown).
    pub fn save_learning(&mut self) {
        self.engine.save_learning();
    }

    /// Handle one request line. Returns the response line to write, or
    /// `None` for notifications (requests without an id).
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        let request: Request = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                let resp = Response::failure(
                    Value::Null,
                    RpcError::new(RpcError::PARSE_ERROR, format!("parse error: {e}")),
                );
                return Some(serde_json::to_string(&resp).expect("response serialization"));
            }
        };

        let id = request.id.clone();
        let result = self.dispatch(&request.method, request.params);
        // A background search only serves the conversion it was started for;
        // once that is committed or cancelled, stop spending CPU on it.
        if !matches!(self.engine.state(), InputState::Conversion { .. }) {
            self.engine.cancel_deep_search();
        }
        // Nothing being typed: the live-conversion context isn't needed until
        // the next composition starts.
        if matches!(self.engine.state(), InputState::Empty) {
            self.engine.release_idle_caches();
        }

        let id = id?; // notification: execute but don't respond
        let resp = match result {
            Ok(value) => Response::success(id, value),
            Err(error) => Response::failure(id, error),
        };
        Some(serde_json::to_string(&resp).expect("response serialization"))
    }

    /// Error response for a request whose handler panicked. The engine is
    /// reset, since the panic may have left it mid-transition.
    pub fn recover_from_panic(&mut self, line: &str) -> Option<String> {
        self.engine.reset();
        let id = serde_json::from_str::<Request>(line).ok()?.id?;
        let resp = Response::failure(
            id,
            RpcError::new(RpcError::INTERNAL_ERROR, "internal error"),
        );
        Some(serde_json::to_string(&resp).expect("response serialization"))
    }

    fn dispatch(&mut self, method: &str, params: Value) -> Result<Value, RpcError> {
        match method {
            "init" => self.handle_init(),
            "process_key" => {
                let params: ProcessKeyParams = parse_params(params)?;
                let event =
                    KeyEvent::new(Keysym(params.keysym), params.modifiers, !params.is_release);
                let result = self.engine.process_key(&event);
                self.key_result(result)
            }
            "select_candidate" => {
                let params: SelectCandidateParams = parse_params(params)?;
                if params.page_index >= CandidateList::DEFAULT_PAGE_SIZE {
                    return Err(RpcError::new(
                        RpcError::INVALID_PARAMS,
                        format!("page_index out of range: {}", params.page_index),
                    ));
                }
                let result = self.engine.select_candidate_on_page(params.page_index);
                self.key_result(result)
            }
            "deepen_inference" => {
                let result = self.engine.deepen_inference();
                self.key_result(result)
            }
            "forget_candidate" => {
                let params: SelectCandidateParams = parse_params(params)?;
                if params.page_index >= CandidateList::DEFAULT_PAGE_SIZE {
                    return Err(RpcError::new(
                        RpcError::INVALID_PARAMS,
                        format!("page_index out of range: {}", params.page_index),
                    ));
                }
                let result = self.engine.forget_candidate_on_page(params.page_index);
                self.key_result(result)
            }
            "commit" => {
                let result = self.engine.commit_result();
                self.key_result(result)
            }
            "reset" => {
                self.engine.reset();
                Ok(json!({}))
            }
                "set_surrounding_text" => {
                let params: SurroundingTextParams = parse_params(params)?;
                self.engine
                    .set_surrounding_text_at(&params.text, params.cursor_pos);
                Ok(json!({}))
            }
            "set_secure_input" => {
                let params: SecureInputParams = parse_params(params)?;
                self.engine.set_secure_input(params.active);
                Ok(json!({}))
            }

            "save_learning" => {
                self.engine.save_learning();
                Ok(json!({}))
            }
            "status" => {
                let state = match self.engine.state() {
                    InputState::Empty => "empty",
                    InputState::Composing { .. } => "composing",
                    InputState::Conversion { .. } => "conversion",
                };
                serde_json::to_value(StatusResult {
                    initialized: self.initialized,
                    model_name: self.engine.model_name(),
                    state,
                })
                .map_err(internal_error)
            }
            other => Err(RpcError::new(
                RpcError::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            )),
        }
    }

    fn handle_init(&mut self) -> Result<Value, RpcError> {
        if !self.initialized {
            let settings = self
                .settings
                .take()
                .unwrap_or_else(|| Settings::load().unwrap_or_default());
            if let Err(e) = self.engine.init_from_settings(&settings) {
                // Keep the settings so a retried `init` uses the same ones.
                self.settings = Some(settings);
                return Err(RpcError::new(RpcError::INIT_FAILED, format!("{e:#}")));
            }
            self.initialized = true;
        }
        serde_json::to_value(InitResult {
            protocol_version: PROTOCOL_VERSION,
            model_name: self.engine.model_name(),
        })
        .map_err(internal_error)
    }

    /// Whether deep-search work is scheduled or in flight.
    pub fn deep_search_pending(&self) -> bool {
        self.engine.deep_search_pending()
    }

    /// Hand any scheduled deep search out to be run on the worker thread.
    ///
    /// Cheap and non-blocking — it only builds the task list. The model work
    /// happens on the worker, which is what keeps key events from ever waiting
    /// behind a wide beam.
    pub fn take_deep_job(&mut self) -> Option<DeepSearchJob> {
        self.engine.take_deep_job()
    }

    /// Merge a finished deep search and render the `update_candidates`
    /// notification. `None` when the outcome is stale or added nothing.
    pub fn apply_deep_outcome(&mut self, outcome: DeepSearchOutcome) -> Option<String> {
        let result = self.engine.apply_deep_outcome(outcome)?;
        Some(self.notification(result))
    }

    /// Wrap an engine result as a server-initiated `update_candidates` line.
    fn notification(&self, result: EngineResult) -> String {
        let actions: Vec<Action> = result.actions.into_iter().map(to_action).collect();
        json!({
            "jsonrpc": "2.0",
            "method": "update_candidates",
            "params": { "actions": actions },
        })
        .to_string()
    }

    fn key_result(&self, result: EngineResult) -> Result<Value, RpcError> {
        let actions = result.actions.into_iter().map(to_action).collect();
        serde_json::to_value(KeyResult {
            consumed: result.consumed,
            actions,
            conversion_ms: self.engine.last_conversion_ms(),
            process_key_ms: self.engine.last_process_key_ms(),
        })
        .map_err(internal_error)
    }
}

fn parse_params<T: serde::de::DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    serde_json::from_value(params)
        .map_err(|e| RpcError::new(RpcError::INVALID_PARAMS, format!("invalid params: {e}")))
}

fn internal_error(e: serde_json::Error) -> RpcError {
    RpcError::new(RpcError::INTERNAL_ERROR, format!("internal error: {e}"))
}

fn to_action(action: EngineAction) -> Action {
    use crate::core::preedit::AttributeType;

    match action {
        EngineAction::UpdatePreedit(preedit) => Action::UpdatePreedit {
            caret: preedit.caret(),
            attributes: preedit
                .attributes()
                .iter()
                .map(|a| PreeditAttr {
                    start: a.start,
                    end: a.end,
                    style: match a.attr_type {
                        AttributeType::Underline => "underline",
                        AttributeType::UnderlineDouble => "underline_double",
                        AttributeType::Highlight => "highlight",
                        AttributeType::Reverse => "reverse",
                    },
                })
                .collect(),
            text: preedit.text().to_string(),
        },
        EngineAction::ShowCandidates(list) => Action::ShowCandidates {
            candidates: list
                .page_candidates()
                .iter()
                .map(|c| CandidateItem {
                    // The source label is how the engine tags a learning-cache
                    // candidate; the frontend only needs the yes/no.
                    is_learned: c.source_label.as_deref()
                        == Some(crate::core::engine::LEARNING_SOURCE_LABEL),
                    text: c.text.clone(),
                    description: c.description.clone().filter(|s| !s.is_empty()),
                })
                .collect(),
            cursor: list.page_cursor(),
            page: list.current_page(),
            total_pages: list.total_pages(),
        },
        EngineAction::HideCandidates => Action::HideCandidates,
        EngineAction::Commit(text) => Action::Commit { text },
        EngineAction::DeleteSurrounding(text) => Action::DeleteSurrounding { text },
        EngineAction::UpdateAuxText(text) => {
            if text.is_empty() {
                Action::HideAux
            } else {
                Action::UpdateAux {
                    text,
                    button: false,
                }
            }
        }
        EngineAction::UpdateAuxButton(text) => Action::UpdateAux { text, button: true },
        EngineAction::HideAuxText => Action::HideAux,
    }
}

#[cfg(test)]
mod tests;
