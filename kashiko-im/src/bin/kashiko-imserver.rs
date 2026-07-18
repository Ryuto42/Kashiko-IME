//! kashiko-imserver: stdio JSON-RPC engine server for the macOS frontend.
//!
//! Reads newline-delimited JSON-RPC 2.0 requests from stdin and writes one
//! response per line to stdout. Logs go to stderr (`RUST_LOG` controls the
//! filter; defaults to `info`). The learning cache is saved on EOF, so the
//! frontend should close the child's stdin (or send `save_learning`) before
//! terminating it.
//!
//! `--prefetch-models` downloads every conversion model listed in
//! `models.toml` into the HuggingFace cache and exits (used by `make install`
//! to avoid a multi-minute download on first launch).

use std::io::{BufRead, Write};

use kashiko_im::server::ImServer;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    if std::env::args().any(|arg| arg == "--prefetch-models") {
        if let Err(e) = kashiko_engine::kanji::hf_download::prefetch_all_models() {
            tracing::error!("model prefetch failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    let mut server = ImServer::new();
    let mut stdout = std::io::stdout().lock();

    // stdin is read on its own thread and fed through a channel so the main
    // loop can interleave deep-search steps with request handling: while a
    // search is pending, each iteration first drains any waiting request
    // (keys stay responsive — worst case they wait one model call), and only
    // steps the search when the channel is idle.
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin().lock();
        for line in stdin.lines() {
            match line {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("stdin read error: {e}");
                    break;
                }
            }
        }
    });

    tracing::info!("kashiko-imserver started (pid={})", std::process::id());

    'main: loop {
        let line = if server.deep_search_pending() {
            match rx.try_recv() {
                Ok(line) => Some(line),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No request waiting: advance the search by one step.
                    // The step cancels itself if the conversion was already
                    // committed, so no time is wasted on finished input.
                    if let Some(note) = server.step_deep_search()
                        && writeln!(stdout, "{note}")
                            .and_then(|_| stdout.flush())
                            .is_err()
                    {
                        break 'main;
                    }
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(line) => Some(line),
                Err(_) => break, // stdin closed: frontend is gone
            }
        };

        let Some(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = server.handle_line(&line)
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
