//! Encryption for the learning cache on disk.
//!
//! The cache is a record of what the user has typed, so it should not sit on
//! disk as readable text where a backup, a sync client, or anything else that
//! walks the home directory picks it up. The file is written as
//! ChaCha20-Poly1305 (authenticated, so a corrupted or tampered file fails
//! loudly instead of loading as garbage) under a key generated once per install.
//!
//! What this does and does not buy: the key lives beside the cache in a
//! `0600` file, so anything already running as the user can read both. It is
//! protection against the file being *read in passing* — copied out by a
//! backup, indexed, opened by accident — not against an attacker with the
//! account. Storing the key in the login keychain instead would raise that bar,
//! and needs the Swift side to hand a key down at startup.
//!
//! Existing plaintext caches are read as-is and rewritten encrypted on the next
//! save, so upgrading loses nothing.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey};
use ring::rand::{SecureRandom, SystemRandom};

/// Marks an encrypted cache. Plaintext caches start with `#` (their comment
/// header) and so can never be mistaken for one.
const MAGIC: &[u8] = b"KSHKLC1\n";

const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;

/// Where the key for `cache_path` lives: alongside it, same stem, `.key`.
pub fn key_path(cache_path: &Path) -> PathBuf {
    cache_path.with_extension("key")
}

/// Read the install's key, generating and storing one if there isn't one yet.
fn load_or_create_key(cache_path: &Path) -> Result<[u8; KEY_LEN]> {
    let path = key_path(cache_path);
    if let Ok(bytes) = std::fs::read(&path)
        && bytes.len() == KEY_LEN
    {
        let mut key = [0u8; KEY_LEN];
        key.copy_from_slice(&bytes);
        return Ok(key);
    }
    let mut key = [0u8; KEY_LEN];
    SystemRandom::new()
        .fill(&mut key)
        .map_err(|_| anyhow::anyhow!("no system randomness available"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_private(&path, &key).with_context(|| format!("writing key to {path:?}"))?;
    Ok(key)
}

/// Write `bytes` to `path` readable only by the current user.
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(bytes)?;
    file.flush()?;
    Ok(())
}

fn sealing_key(key: &[u8; KEY_LEN]) -> Result<LessSafeKey> {
    let unbound = UnboundKey::new(&aead::CHACHA20_POLY1305, key)
        .map_err(|_| anyhow::anyhow!("bad learning-cache key"))?;
    Ok(LessSafeKey::new(unbound))
}

/// Encrypt `plaintext` and write it to `path`, creating the key if needed.
pub fn write_encrypted(path: &Path, plaintext: &str) -> Result<()> {
    let key = load_or_create_key(path)?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    SystemRandom::new()
        .fill(&mut nonce_bytes)
        .map_err(|_| anyhow::anyhow!("no system randomness available"))?;

    let mut buf = plaintext.as_bytes().to_vec();
    sealing_key(&key)?
        .seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce_bytes),
            Aad::empty(),
            &mut buf,
        )
        .map_err(|_| anyhow::anyhow!("encrypting the learning cache failed"))?;

    let mut out = Vec::with_capacity(MAGIC.len() + NONCE_LEN + buf.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&buf);
    write_private(path, &out)
}

/// Read `path` as text, decrypting it when it is an encrypted cache.
///
/// A file without the magic header is returned as-is: that is a cache written
/// before encryption existed, and it is rewritten encrypted on the next save.
pub fn read_maybe_encrypted(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(MAGIC) {
        return Ok(String::from_utf8_lossy(&bytes).into_owned());
    }
    let body = &bytes[MAGIC.len()..];
    if body.len() < NONCE_LEN {
        bail!("learning cache is truncated");
    }
    let (nonce_bytes, ciphertext) = body.split_at(NONCE_LEN);
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(nonce_bytes);

    let key = load_or_create_key(path)?;
    let mut buf = ciphertext.to_vec();
    let plaintext = sealing_key(&key)?
        .open_in_place(
            Nonce::assume_unique_for_key(nonce),
            Aad::empty(),
            &mut buf,
        )
        .map_err(|_| {
            anyhow::anyhow!("the learning cache could not be decrypted (wrong or missing key)")
        })?;
    Ok(String::from_utf8_lossy(plaintext).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kashiko-lf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_round_trip_returns_the_same_text() {
        let path = temp_dir().join("round-trip.tsv");
        let text = "# header\nてんぷれーと\ttemplate\t3\t1700000000\n";
        write_encrypted(&path, text).unwrap();
        assert_eq!(read_maybe_encrypted(&path).unwrap(), text);
    }

    #[test]
    fn the_file_on_disk_is_not_readable_text() {
        let path = temp_dir().join("opaque.tsv");
        write_encrypted(&path, "ひみつのぶんしょう\t秘密の文章\t1\t1700000000\n").unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert!(
            !String::from_utf8_lossy(&raw).contains("秘密の文章"),
            "the surface is still readable on disk"
        );
    }

    #[test]
    fn a_plaintext_cache_still_loads() {
        // Upgrading must not lose what the user has already taught the IME.
        let path = temp_dir().join("legacy.tsv");
        let text = "# kashiko learning cache v1\nあ\t亜\t1\t1700000000\n";
        std::fs::write(&path, text).unwrap();
        assert_eq!(read_maybe_encrypted(&path).unwrap(), text);
    }

    #[test]
    fn tampering_is_detected() {
        let path = temp_dir().join("tampered.tsv");
        write_encrypted(&path, "あ\t亜\t1\t1700000000\n").unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_maybe_encrypted(&path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn the_key_is_private_to_the_user() {
        use std::os::unix::fs::PermissionsExt;
        let path = temp_dir().join("perms.tsv");
        write_encrypted(&path, "あ\t亜\t1\t1700000000\n").unwrap();
        for p in [key_path(&path), path] {
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{p:?} is {mode:o}");
        }
    }
}
