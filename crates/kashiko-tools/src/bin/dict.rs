use anyhow::Result;
use clap::{Parser, Subcommand};
use kashiko_engine::dict::Dictionary;
use std::path::PathBuf;

/// Kashiko dictionary tool — build and search dictionaries.
#[derive(Parser, Debug)]
#[command(name = "kashiko-dict")]
#[command(about = "Kashiko dictionary tool — build and search dictionaries")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build a binary dictionary from JSON or Mozc TSV format.
    ///
    /// Supports two input formats:
    /// - `json`: Array of {reading, candidates: [{surface, score}]}
    /// - `mozc`: Mozc/Google IME TSV (reading\tword\tPOS\tcomment)
    ///
    /// Format is auto-detected from file extension (.json → JSON, otherwise → Mozc TSV),
    /// or can be explicitly specified with --format.
    Build {
        /// Input dictionary file (JSON or Mozc TSV)
        input: PathBuf,

        /// Output binary dictionary file
        #[arg(short, long, default_value = "dict.bin")]
        output: PathBuf,

        /// Input format: json or mozc (auto-detected from extension if omitted)
        #[arg(short, long)]
        format: Option<String>,
    },

    /// Search dictionaries and print `reading<TAB>surface<TAB>score` lines.
    Search {
        /// Dictionary files to load (KRKN binary or Mozc TSV, auto-detected)
        #[arg(required = true)]
        dicts: Vec<PathBuf>,

        /// Reading (or, with --surface, surface text) to look up
        #[arg(short, long, required_unless_present = "all")]
        query: Option<String>,

        /// Search by surface form instead of reading
        #[arg(short, long)]
        surface: bool,

        /// Use prefix search instead of exact match (reading search only)
        #[arg(short, long)]
        prefix: bool,

        /// Show all entries (dump entire dictionary)
        #[arg(short, long)]
        all: bool,
    },
}

// --- build subcommand ---

fn run_build(input: PathBuf, output: PathBuf, format: Option<String>) -> Result<()> {
    let format =
        format
            .as_deref()
            .unwrap_or_else(|| match input.extension().and_then(|e| e.to_str()) {
                Some("json") => "json",
                _ => "mozc",
            });

    eprintln!(
        "Building dictionary from {:?} (format: {})...",
        input, format
    );

    let dict = match format {
        "json" => Dictionary::build_from_json(&input)?,
        "mozc" => Dictionary::build_from_mozc_tsv(&input)?,
        other => anyhow::bail!("Unknown format: {}. Use 'json' or 'mozc'.", other),
    };

    eprintln!("Saving to {:?}...", output);
    dict.save(&output)?;

    eprintln!("Done.");
    Ok(())
}

// --- search subcommand ---

fn run_search(
    dicts: Vec<PathBuf>,
    query: Option<String>,
    surface: bool,
    prefix: bool,
    all: bool,
) -> Result<()> {
    eprintln!("Loading dictionaries...");
    let mut loaded = Vec::new();
    for path in &dicts {
        eprintln!("  Loading {:?}...", path);
        let dict = Dictionary::load_auto(path)?;
        loaded.push(dict);
    }

    let dict = if loaded.len() == 1 {
        // Safety: len() == 1 guarantees next() returns Some
        loaded.into_iter().next().expect("single dictionary loaded")
    } else {
        Dictionary::merge(loaded)?.expect("at least one dictionary loaded")
    };

    eprintln!("Dictionary loaded.");

    if all {
        let mut stdout = std::io::stdout().lock();
        let count = dict.dump_all(&mut stdout)?;
        eprintln!("({} entries total)", count);
        return Ok(());
    }

    if let Some(query) = &query {
        if surface {
            let results = dict.search_by_surface(query);
            if results.is_empty() {
                eprintln!("No entries found with surface containing \"{}\"", query);
            } else {
                for (reading, surface, score) in &results {
                    println!("{}\t{}\t{}", reading, surface, score);
                }
                eprintln!("({} results)", results.len());
            }
        } else if prefix {
            let results = dict.common_prefix_search(query);
            if results.is_empty() {
                eprintln!("No entries found with reading prefix \"{}\"", query);
            } else {
                for result in &results {
                    for cand in &result.candidates {
                        println!("{}\t{}\t{}", result.reading, cand.surface, cand.score);
                    }
                }
                eprintln!("({} readings matched)", results.len());
            }
        } else {
            match dict.exact_match_search(query) {
                Some(result) => {
                    for cand in &result.candidates {
                        println!("{}\t{}\t{}", result.reading, cand.surface, cand.score);
                    }
                    eprintln!("({} candidates)", result.candidates.len());
                }
                None => {
                    eprintln!("No entry found for reading \"{}\"", query);
                }
            }
        }
        return Ok(());
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            input,
            output,
            format,
        } => run_build(input, output, format),
        Commands::Search {
            dicts,
            query,
            surface,
            prefix,
            all,
        } => run_search(dicts, query, surface, prefix, all),
    }
}
