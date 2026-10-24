//! kashiko-imserver: stdio JSON-RPC engine server for the macOS frontend.
//!
//! Reads newline-delimited JSON-RPC 2.0 requests from stdin and writes one
//! response per line to stdout. Logs go to stderr (`RUST_LOG` controls the
//! filter; defaults to `info`). The learning cache is saved on EOF, so the
//! frontend should close the child's stdin (or send `save_learning`) before
//! terminating it.
//!
//! `--prefetch-models` downloads the configured conversion models (main and
//! light) into the HuggingFace cache, builds the mapped dictionary cache, and
//! exits (used by `make install`
//! to avoid a multi-minute download on first launch).

use std::io::{BufRead, Write};

use kashiko_im::server::ImServer;

/// Named so a profiler or crash report shows the thread is Kashiko's.
fn spawn_named(name: &str, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(f)
        .expect("failed to spawn thread");
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    // The default hook prints the panic message, which can quote the text
    // being converted (e.g. a slicing panic shows the whole string).
    std::panic::set_hook(Box::new(|info| {
        tracing::error!("panic at {:?}", info.location());
    }));

    if std::env::args().any(|arg| arg == "--prefetch-models") {
        let settings = kashiko_im::config::Settings::load().unwrap_or_default();
        let default = kashiko_engine::kanji::registry().default_model.clone();
        let main = settings.conversion.model.unwrap_or(default);
        let mut ids = vec![main.as_str()];
        if let Some(light) = settings.conversion.light_model.as_deref() {
            ids.push(light);
        }
        if let Err(e) = kashiko_engine::kanji::hf_download::prefetch_models(&ids) {
            tracing::error!("model prefetch failed: {e}");
            std::process::exit(1);
        }
        // Build the mapped dictionary cache now rather than on the IME's first
        // launch, whose process would keep the memory used to build it.
        let dict = settings
            .conversion
            .dict_path
            .map(std::path::PathBuf::from)
            .or_else(|| kashiko_im::config::Settings::data_dir().map(|d| d.join("dict.bin")));
        if let Some(dict) = dict.filter(|d| d.exists())
            && let Err(e) = kashiko_engine::Dictionary::load_mapped(&dict)
        {
            tracing::warn!("dictionary cache not built: {e}");
        }
        return;
    }

    let mut server = ImServer::new();
    let mut stdout = std::io::stdout().lock();

    // Everything the loop reacts to arrives on one channel, so the loop is
    // *always* blocked on `recv` and never inside a model call. That is the
    // whole point: a wide beam over a long sentence takes seconds, and while it
    // runs the user still has to be able to move through candidates and commit.
    //
    // The previous design interleaved instead — it stepped the search whenever
    // no request was waiting — which left a key event stuck behind whichever
    // task was running. One greedy pass over a 30-character reading is already
    // ~350ms, and the inference button widens the beam from there.
    enum Msg {
        Request(String),
        DeepDone(kashiko_im::DeepSearchOutcome),
        RefineDone(kashiko_im::RefineOutcome),
        /// stdin closed. Sent explicitly: the worker keeps its sender alive,
        /// so `recv` alone would never report the end.
        Eof,
    }

    let (tx, rx) = std::sync::mpsc::channel::<Msg>();

    let stdin_tx = tx.clone();
    spawn_named("Kashiko IME: input", move || {
        let stdin = std::io::stdin().lock();
        for line in stdin.lines() {
            match line {
                Ok(line) => {
                    if stdin_tx.send(Msg::Request(line)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("stdin read error: {e}");
                    break;
                }
            }
        }
        let _ = stdin_tx.send(Msg::Eof);
    });

    // The background worker: deep searches during conversion, live-conversion
    // refinement while composing. It holds its own handle on the (immutable)
    // model and builds its own llama.cpp context, so it needs no lock against
    // the main thread's own inference.
    enum Job {
        Deep(kashiko_im::DeepSearchJob),
        Refine(kashiko_im::RefineJob),
    }
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
    let done_tx = tx.clone();
    spawn_named("Kashiko IME: background inference", move || {
        for job in job_rx {
            let done = match job {
                Job::Deep(job) => Msg::DeepDone(kashiko_im::run_deep_job(job)),
                Job::Refine(job) => Msg::RefineDone(kashiko_im::run_refine_job(job)),
            };
            if done_tx.send(done).is_err() {
                break;
            }
        }
    });
    // Both threads hold a sender; drop ours so `recv` ends when they are gone.
    drop(tx);

    tracing::info!("Kashiko IME Engine started (pid={})", std::process::id());

    loop {
        // A request may have scheduled a search; hand it over before waiting.
        if let Some(job) = server.take_deep_job()
            && job_tx.send(Job::Deep(job)).is_err()
        {
            tracing::error!("background worker is gone");
        }
        if let Some(job) = server.take_refine_job()
            && job_tx.send(Job::Refine(job)).is_err()
        {
            tracing::error!("background worker is gone");
        }

        let Ok(msg) = rx.recv() else { break };
        let line = match msg {
            Msg::DeepDone(outcome) => {
                if let Some(note) = server.apply_deep_outcome(outcome)
                    && writeln!(stdout, "{note}")
                        .and_then(|_| stdout.flush())
                        .is_err()
                {
                    break;
                }
                continue;
            }
            Msg::RefineDone(outcome) => {
                if let Some(note) = server.apply_refine_outcome(outcome)
                    && writeln!(stdout, "{note}")
                        .and_then(|_| stdout.flush())
                        .is_err()
                {
                    break;
                }
                continue;
            }
            Msg::Request(line) => line,
            Msg::Eof => break,
        };

        if line.trim().is_empty() {
            continue;
        }
        let response =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| server.handle_line(&line)))
                .unwrap_or_else(|_| server.recover_from_panic(&line));
        if let Some(response) = response
            && writeln!(stdout, "{response}")
                .and_then(|_| stdout.flush())
                .is_err()
        {
            // stdout closed: frontend is gone
            break;
        }
    }

    tracing::info!("stdin closed, saving learning cache and exiting");
    server.save_learning();
}
