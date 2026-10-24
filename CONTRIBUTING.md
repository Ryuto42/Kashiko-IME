# コントリビューションガイド

## ビルドの前提

- macOS 13 以降（Apple Silicon 推奨）
- Xcode Command Line Tools（`xcode-select --install`）
- Rust 安定版（1.92 以上）
- CMake（`brew install cmake`）
- GNU Make

## ビルドとインストール

リポジトリ直下で実行します。フォルダ構成は [README](README.md#構成) を参照してください。

```sh
make install
```

## テスト

```sh
cargo test --workspace
swift test --package-path macos
```

実モデルを使う `#[ignore]` テストは次のコマンドで実行します。

```sh
cargo test --release -p kashiko-im --lib real_model -- --ignored --test-threads=1
```

注意: このテストは**利用者自身の設定・学習キャッシュ**（`~/Library/Application Support/Kashiko/`）を読み込みます。必要なら事前にバックアップしてください。

`crates/kashiko-engine/tests/kanji_conversion_tests.rs` はモデルを取得できない環境（オフライン）では自動的にスキップされます。

## llama.cpp のビルドに関する注意

Metal を無効化するため、`.cargo/config.toml` と `crates/kashiko-engine/cmake/ggml-cpu-only.cmake` で CMake の設定を上書きしています。これらを変更したときは、次を実行して再ビルドしてください。

```sh
cargo clean -p llama-cpp-sys-2
```

## リリース手順

1. `macos/Info.plist` の `CFBundleShortVersionString` を上げる。
2. `vX.Y.Z` 形式のタグを push する。
3. `.github/workflows/release.yml` がビルドして公開する。
