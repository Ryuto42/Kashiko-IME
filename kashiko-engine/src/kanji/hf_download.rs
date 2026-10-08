//! HuggingFace model download utilities
//!
//! Downloads GGUF models from HuggingFace Hub and caches them locally.
//! Model definitions are loaded from `models.toml` via [`super::model_config`].

use super::error::KanjiError;
use super::model_config::{ModelFamily, VariantConfig, registry};
type Result<T> = super::error::Result<T>;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use ring::digest::{Context, SHA256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Download a file from HuggingFace Hub at a pinned commit and verify its SHA-256
///
/// Returns the local path to the downloaded file.
/// The file is cached in the HuggingFace cache directory (~/.cache/huggingface/hub/).
/// The cached file is re-verified on every call, so a tampered cache is
/// rejected as well.
///
/// # Arguments
/// * `repo_id` - HuggingFace repository ID
/// * `revision` - Commit SHA to download from
/// * `filename` - Filename to download
/// * `sha256` - Expected SHA-256 of the file (lowercase hex)
///
/// # Environment Variables
/// * `HF_TOKEN` - HuggingFace API token (required for private repositories)
pub fn download_gguf(
    repo_id: &str,
    revision: &str,
    filename: &str,
    sha256: &str,
) -> Result<PathBuf> {
    // Check for HF_TOKEN environment variable
    let mut builder = ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|e| KanjiError::Download(e.into()))?;

    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    tracing::info!("Downloading {} from {}@{}...", filename, repo_id, revision);

    let path = repo
        .get(filename)
        .map_err(|e| KanjiError::Download(e.into()))?;

    verify_sha256(&path, sha256)?;

    tracing::info!("Downloaded to {:?}", path);

    Ok(path)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let mut file = std::fs::File::open(path).map_err(|e| KanjiError::Download(e.into()))?;
    let mut ctx = Context::new(&SHA256);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| KanjiError::Download(e.into()))?;
        if n == 0 {
            break;
        }
        ctx.update(&buf[..n]);
    }
    let actual: String = ctx
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(KanjiError::Download(
            format!(
                "sha256 mismatch for {}: expected {}, got {}",
                path.display(),
                expected,
                actual
            )
            .into(),
        ));
    }
    Ok(())
}

/// Get local path for a variant, downloading if not cached.
pub fn get_variant_path(family: &ModelFamily, variant: &VariantConfig) -> Result<PathBuf> {
    download_gguf(
        &family.repo_id,
        &family.revision,
        &variant.filename,
        &variant.sha256,
    )
}

/// Get the local path to `tokenizer.json` for a model family, downloading if necessary.
pub fn get_tokenizer_path(family: &ModelFamily) -> Result<PathBuf> {
    download_gguf(
        &family.repo_id,
        &family.revision,
        "tokenizer.json",
        &family.tokenizer_sha256,
    )
}

/// Convenience: look up a variant id in the global registry and return its local GGUF path.
pub fn get_path_by_id(variant_id: &str) -> Result<PathBuf> {
    let (family, variant) = registry()
        .find_variant(variant_id)
        .ok_or_else(|| KanjiError::UnknownVariant(variant_id.to_string()))?;
    get_variant_path(family, variant)
}

/// Convenience: look up a variant id and return the tokenizer path.
pub fn get_tokenizer_path_by_id(variant_id: &str) -> Result<PathBuf> {
    let (family, _variant) = registry()
        .find_variant(variant_id)
        .ok_or_else(|| KanjiError::UnknownVariant(variant_id.to_string()))?;
    get_tokenizer_path(family)
}

/// Download the given model variants (GGUF + tokenizer) into the HuggingFace
/// cache, without loading any of them.
///
/// Used by `kashiko-imserver --prefetch-models` so installers can warm the
/// cache up front instead of blocking the first conversion on a download.
pub fn prefetch_models(variant_ids: &[&str]) -> Result<()> {
    for id in variant_ids {
        let (family, variant) = registry()
            .find_variant(id)
            .ok_or_else(|| KanjiError::UnknownVariant(id.to_string()))?;
        let model_path = get_variant_path(family, variant)?;
        let tokenizer_path = get_tokenizer_path(family)?;
        tracing::info!(
            "Model '{}' ready: {} (tokenizer: {})",
            variant.id,
            model_path.display(),
            tokenizer_path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_default_variant_exists() {
        let reg = registry();
        let (family, variant) = reg.default_variant().expect("default variant must exist");
        assert!(!family.repo_id.is_empty());
        assert!(!variant.filename.is_empty());
    }

    #[test]
    fn test_find_all_variants() {
        let reg = registry();
        for id in reg.all_variant_ids() {
            assert!(
                reg.find_variant(id).is_some(),
                "find_variant should succeed for '{}'",
                id
            );
        }
    }

    #[test]
    fn test_registry_entries_are_pinned() {
        let is_hex = |s: &str, len: usize| {
            s.len() == len && s.chars().all(|c| c.is_ascii_hexdigit())
        };
        for family in registry().models.values() {
            assert!(is_hex(&family.revision, 40), "{}", family.repo_id);
            assert!(is_hex(&family.tokenizer_sha256, 64), "{}", family.repo_id);
            for variant in family.variants.values() {
                assert!(is_hex(&variant.sha256, 64), "{}", variant.id);
            }
        }
    }

    #[test]
    fn test_verify_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::write(&path, b"abc").unwrap();
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify_sha256(&path, abc).is_ok());
        assert!(verify_sha256(&path, &"0".repeat(64)).is_err());
    }

    #[test]
    fn test_get_path_by_id_unknown() {
        let result = get_path_by_id("nonexistent-model-id");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_path_by_id_unknown() {
        let result = get_tokenizer_path_by_id("nonexistent-model-id");
        assert!(result.is_err());
    }
}
