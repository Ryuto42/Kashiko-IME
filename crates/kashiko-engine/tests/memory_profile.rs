//! Where the resident memory goes. Ignored by default (loads the real
//! dictionary and models).
//!
//!     cargo test -p kashiko-engine --release --test memory_profile -- --ignored --nocapture

fn init_log() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("debug"))
        .try_init();
}

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
        / 1024.0
}

fn dict_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap();
    std::path::Path::new(&home).join("Library/Application Support/Kashiko/dict.bin")
}

#[test]
#[ignore]
fn where_the_memory_goes() {
    init_log();
    let base = rss_mb();
    println!("{:<40} {:8.1} MB", "baseline", base);

    let file_mb = std::fs::metadata(dict_path())
        .map(|m| m.len() as f64 / 1048576.0)
        .unwrap_or(0.0);
    let t = std::time::Instant::now();
    let d = kashiko_engine::Dictionary::load(dict_path()).expect("dict");
    let load = t.elapsed();
    let after_dict = rss_mb();
    println!(
        "{:<40} {:8.1} MB   (file {:.1} MB, {} readings, load {:.2}s)",
        "+ system dictionary",
        after_dict - base,
        file_mb,
        d.entry_count(),
        load.as_secs_f64()
    );

    let backend = kashiko_engine::Backend::from_variant_id("jinen-v1-small-q5").expect("backend");
    let main = kashiko_engine::KanaKanjiConverter::new(backend).expect("main model");
    let after_main = rss_mb();
    println!(
        "{:<40} {:8.1} MB",
        "+ main model (jinen-v1-small-q5)",
        after_main - after_dict
    );

    let lb = kashiko_engine::Backend::from_variant_id("jinen-v1-xsmall-q5").expect("backend");
    let light = kashiko_engine::KanaKanjiConverter::new(lb).expect("light model");
    let after_light = rss_mb();
    println!(
        "{:<40} {:8.1} MB",
        "+ light model (jinen-v1-xsmall-q5)",
        after_light - after_main
    );

    let _ = main.convert("かんじ", "", 3);
    let _ = light.convert("かんじ", "", 3);
    let after_infer = rss_mb();
    println!(
        "{:<40} {:8.1} MB",
        "+ inference contexts",
        after_infer - after_light
    );

    let _ = kashiko_engine::symbol_description("、");
    let after_sym = rss_mb();
    println!(
        "{:<40} {:8.1} MB",
        "+ symbol table",
        after_sym - after_infer
    );

    use kashiko_engine::rewriter::Rewriter;
    let emoji = kashiko_engine::rewriter::EmojiRewriter::new();
    let _ = emoji.rewrite(":smile");
    let after_emoji = rss_mb();
    println!("{:<40} {:8.1} MB", "+ emoji table", after_emoji - after_sym);

    let _ = after_emoji;

    println!("{:<40} {:8.1} MB", "TOTAL", after_emoji);
    drop(d);
}
