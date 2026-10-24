//! Settings configuration
//!
//! Manages user-configurable settings for the IME.
//! Default values are defined in `config/default.toml`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use directories::BaseDirs;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Default configuration TOML embedded from config/default.toml
const DEFAULT_CONFIG_TOML: &str = include_str!("../../config/default.toml");

/// Configuration settings for the IME
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Conversion settings
    pub conversion: ConversionSettings,
    /// Learning cache settings
    pub learning: LearningSettings,
}

/// Conversion strategy mode
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrategyMode {
    /// Adaptive: dynamically switch between main and light models based on latency
    #[default]
    Adaptive,
    /// Light: use light_model only (loaded into main slot, beam search on Space)
    Light,
    /// Main: use main model only (no light model loaded)
    Main,
}

/// Conversion-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionSettings {
    /// Conversion strategy mode (adaptive, light, main)
    pub strategy: StrategyMode,
    /// Number of candidates to show on Space conversion
    pub num_candidates: usize,
    /// Use surrounding text (text left of cursor) as context for conversion
    pub use_context: bool,
    /// Maximum number of surrounding text characters passed to the conversion API
    pub max_context_length: usize,
    /// Maximum reading length (in characters) converted by the model in a single
    /// call during live conversion. The composing buffer is split into chunks
    /// of at most this many characters so per-keystroke latency stays bounded
    /// for long input; each chunk's left context is the converted text of the
    /// preceding chunks.
    pub composing_chunk_len: usize,
    /// Path to dictionary binary file (optional, defaults to data_dir/dict.bin)
    pub dict_path: Option<String>,
    /// Model variant id (optional, defaults to registry default)
    pub model: Option<String>,
    /// Beam search model variant id (used on Space conversion, default model if unset)
    pub light_model: Option<String>,
    /// Token count threshold for beam search (at or below → beam, above → greedy)
    pub short_input_threshold: usize,
    /// Beam width for short input
    pub beam_width: usize,
    /// Maximum acceptable latency in milliseconds for auto-suggest (0 = disabled)
    /// When a main model conversion exceeds this, the engine adaptively switches to light_model
    pub max_latency_ms: u64,
    /// Number of threads for llama.cpp inference (0 = all cores, llama.cpp default)
    pub n_threads: u32,
    /// Enable live conversion at startup (Ctrl+Shift+L still toggles at runtime)
    pub live_conversion: bool,
}

/// Learning cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSettings {
    /// Whether learning is enabled
    pub enabled: bool,
    /// Maximum number of total entries in the learning cache
    pub max_entries: usize,
}

impl Default for Settings {
    fn default() -> Self {
        toml::from_str(DEFAULT_CONFIG_TOML).expect("embedded default.toml must be valid")
    }
}

/// Recursively merge `overlay` TOML values on top of `base`.
fn merge_toml(base: &mut toml::Value, overlay: &toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) => {
            for (key, value) in overlay_table {
                if let Some(base_value) = base_table.get_mut(key) {
                    merge_toml(base_value, value);
                } else {
                    base_table.insert(key.clone(), value.clone());
                }
            }
        }
        (base, _) => {
            *base = overlay.clone();
        }
    }
}

/// Parse user TOML content merged on top of default.toml.
fn parse_with_defaults(user_content: &str) -> Result<Settings> {
    let mut base: toml::Value = toml::from_str(DEFAULT_CONFIG_TOML)?;
    let user: toml::Value = toml::from_str(user_content)?;
    merge_toml(&mut base, &user);
    let settings: Settings = base.try_into()?;
    Ok(settings)
}

/// `~/Library/Application Support/Kashiko`: settings, dictionaries and the
/// learning cache. Earlier versions used `com.kashiko.kashiko-im`, which is
/// moved here the first time the new location is asked for.
fn app_dir() -> Option<PathBuf> {
    Some(resolve_app_dir(BaseDirs::new()?.data_dir()))
}

fn resolve_app_dir(base: &Path) -> PathBuf {
    let dir = base.join("Kashiko");
    let legacy = base.join("com.kashiko.kashiko-im");
    if !legacy.is_dir() {
        return dir;
    }
    if !dir.exists() {
        if let Err(e) = fs::rename(&legacy, &dir) {
            warn!(
                "could not move {} to {}: {e}",
                legacy.display(),
                dir.display()
            );
            return legacy;
        }
        return dir;
    }
    // Both exist: an older engine still running during an update saved its
    // learning cache to the old place after the move. Keep whichever cache is
    // newer — the cache and its key together, as one is useless without the
    // other — and drop the old directory once it is empty.
    let modified = |d: &Path| {
        fs::metadata(d.join("learning.tsv"))
            .and_then(|m| m.modified())
            .ok()
    };
    if let Some(old) = modified(&legacy)
        && modified(&dir).is_none_or(|new| old > new)
        && legacy.join("learning.key").exists()
    {
        for name in ["learning.key", "learning.tsv"] {
            if let Err(e) = fs::rename(legacy.join(name), dir.join(name)) {
                warn!("could not move {name} from {}: {e}", legacy.display());
            }
        }
    }
    for name in ["learning.key", "learning.tsv"] {
        let _ = fs::remove_file(legacy.join(name));
    }
    let _ = fs::remove_dir(&legacy);
    dir
}

impl Settings {
    /// Get the data directory path
    pub fn data_dir() -> Option<PathBuf> {
        app_dir()
    }

    /// Get the configuration directory path
    pub fn config_dir() -> Option<PathBuf> {
        app_dir()
    }

    /// Get the configuration file path
    pub fn config_file() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("config.toml"))
    }

    /// Get the user dictionary directory path.
    ///
    /// All files in this directory are automatically loaded as user dictionaries.
    /// Default (macOS): `~/Library/Application Support/Kashiko/user_dicts/`
    pub fn user_dict_dir() -> Option<PathBuf> {
        Self::data_dir().map(|dir| dir.join("user_dicts"))
    }

    /// Get the learning cache file path.
    ///
    /// Default (macOS): `~/Library/Application Support/Kashiko/learning.tsv`
    pub fn learning_file() -> Option<PathBuf> {
        Self::data_dir().map(|dir| dir.join("learning.tsv"))
    }

    /// Load settings from the default configuration file.
    /// Falls back to embedded default.toml if the config file does not exist.
    pub fn load() -> Result<Self> {
        let Some(config_file) = Self::config_file() else {
            warn!("Could not determine config directory, using defaults");
            return Ok(Self::default());
        };

        if !config_file.exists() {
            debug!("Config file not found, using defaults");
            return Ok(Self::default());
        }

        debug!("Loading config from {:?}", config_file);
        let content = fs::read_to_string(&config_file)?;
        parse_with_defaults(&content)
    }

    /// Load settings from a specific file, merged on top of defaults.
    pub fn load_from(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        parse_with_defaults(&content)
    }

    /// Save settings to the default configuration file
    pub fn save(&self) -> Result<()> {
        let Some(config_file) = Self::config_file() else {
            anyhow::bail!("Could not determine config directory");
        };
        debug!("Saving config to {:?}", config_file);
        self.save_to(&config_file)
    }

    /// Save settings to a specific file (temp file + rename, so a crash never
    /// leaves a half-written config).
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
        tmp_name.push(".tmp");
        let tmp = path.with_file_name(tmp_name);
        fs::write(&tmp, content)?;
        if let Err(e) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }

    /// Check the values the settings screen can change.
    pub fn validate(&self) -> Result<()> {
        let c = &self.conversion;
        if !(1..=20).contains(&c.num_candidates) {
            anyhow::bail!("num_candidates must be 1..=20: {}", c.num_candidates);
        }
        if c.n_threads > 16 {
            anyhow::bail!("n_threads must be 0..=16: {}", c.n_threads);
        }
        if c.max_context_length > 256 {
            anyhow::bail!(
                "max_context_length must be 0..=256: {}",
                c.max_context_length
            );
        }
        if !(100..=100_000).contains(&self.learning.max_entries) {
            anyhow::bail!(
                "learning.max_entries must be 100..=100000: {}",
                self.learning.max_entries
            );
        }
        let registry = kashiko_engine::kanji::registry();
        for (name, id) in [("model", &c.model), ("light_model", &c.light_model)] {
            if let Some(id) = id
                && registry.find_variant(id).is_none()
            {
                anyhow::bail!("unknown {name}: {id}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod app_dir_tests {
    use super::*;

    fn write(dir: &Path, text: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("learning.tsv"), text).unwrap();
        fs::write(dir.join("learning.key"), format!("key-{text}")).unwrap();
    }

    #[test]
    fn the_old_directory_is_moved() {
        let base = tempfile::tempdir().unwrap();
        write(&base.path().join("com.kashiko.kashiko-im"), "old");
        let dir = resolve_app_dir(base.path());
        assert_eq!(fs::read_to_string(dir.join("learning.tsv")).unwrap(), "old");
        assert!(!base.path().join("com.kashiko.kashiko-im").exists());
    }

    #[test]
    fn a_newer_cache_saved_to_the_old_place_is_kept() {
        let base = tempfile::tempdir().unwrap();
        write(&base.path().join("Kashiko"), "moved");
        std::thread::sleep(std::time::Duration::from_millis(20));
        write(&base.path().join("com.kashiko.kashiko-im"), "saved-after");
        let dir = resolve_app_dir(base.path());
        assert_eq!(
            fs::read_to_string(dir.join("learning.tsv")).unwrap(),
            "saved-after"
        );
        assert_eq!(
            fs::read_to_string(dir.join("learning.key")).unwrap(),
            "key-saved-after"
        );
        assert!(!base.path().join("com.kashiko.kashiko-im").exists());
    }

    #[test]
    fn an_older_leftover_is_discarded() {
        let base = tempfile::tempdir().unwrap();
        write(&base.path().join("com.kashiko.kashiko-im"), "stale");
        std::thread::sleep(std::time::Duration::from_millis(20));
        write(&base.path().join("Kashiko"), "current");
        let dir = resolve_app_dir(base.path());
        assert_eq!(
            fs::read_to_string(dir.join("learning.tsv")).unwrap(),
            "current"
        );
        assert!(!base.path().join("com.kashiko.kashiko-im").exists());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.conversion.num_candidates, 9);
        assert!(settings.conversion.use_context);
        assert_eq!(settings.conversion.max_context_length, 32);
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let toml_str = toml::to_string(&settings).unwrap();
        let loaded: Settings = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            loaded.conversion.num_candidates,
            settings.conversion.num_candidates
        );
    }

    #[test]
    fn test_load_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
num_candidates = 5
use_context = false
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.num_candidates, 5);
        assert!(!settings.conversion.use_context);
    }

    #[test]
    fn test_save_to_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");
        let mut settings = Settings::default();
        settings.conversion.num_candidates = 4;
        settings.learning.enabled = false;
        settings.save_to(&path).unwrap();
        let loaded = Settings::load_from(&path).unwrap();
        assert_eq!(loaded.conversion.num_candidates, 4);
        assert!(!loaded.learning.enabled);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn test_validate() {
        assert!(Settings::default().validate().is_ok());
        let mut s = Settings::default();
        s.conversion.num_candidates = 21;
        assert!(s.validate().is_err());
        let mut s = Settings::default();
        s.conversion.n_threads = 17;
        assert!(s.validate().is_err());
        let mut s = Settings::default();
        s.learning.max_entries = 99;
        assert!(s.validate().is_err());
        let mut s = Settings::default();
        s.conversion.model = Some("no-such-model".into());
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_user_dict_dir() {
        let dir = Settings::user_dict_dir();
        // Should return Some on systems with a home directory
        if let Some(dir) = dir {
            assert!(dir.ends_with("user_dicts"));
        }
    }

    #[test]
    fn test_partial_config() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
num_candidates = 3
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.num_candidates, 3);
        // Should use default for unspecified values
        assert!(settings.conversion.use_context);
        assert_eq!(settings.conversion.max_context_length, 32);
    }

    #[test]
    fn test_strategy_default_when_unspecified() {
        // When strategy is not specified, it should default to Adaptive
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
num_candidates = 5
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.strategy, StrategyMode::Adaptive);
    }

    #[test]
    fn test_strategy_light() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
strategy = "light"
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.strategy, StrategyMode::Light);
    }

    #[test]
    fn test_strategy_main() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
strategy = "main"
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.strategy, StrategyMode::Main);
    }
}
