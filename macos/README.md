# macos

macOS 向け Kashiko 日本語入力（InputMethodKit + Swift）。

## アーキテクチャ

サーバー・クライアント構成です。IMEの状態機械・ローマ字変換・かな漢字変換はすべて Rust 側（`kashiko-im`）にあり、Swift 側は InputMethodKit との橋渡しに徹します。

```
┌───────────────────────────────┐                            ┌─────────────────────────────┐
│ Kashiko IME (Swift)           │  JSON-RPC 2.0 (改行区切り)  │ Kashiko IME Engine (Rust)   │
│ ・IMKServer / InputController │ ◀────────────────────────▶ │ ・InputMethodEngine         │
│ ・NSEvent → XKB keysym変換    │    stdin/stdout パイプ      │ ・ローマ字変換・状態機械    │
│ ・preedit / 候補ウィンドウ描画 │                            │ ・llama.cpp推論・辞書・学習 │
│ ・子プロセス管理・自動再起動   │                            │                             │
└───────────────────────────────┘                            └─────────────────────────────┘
        どちらも Kashiko.app バンドル内 (Contents/MacOS/)
```

- プロトコル定義: `crates/kashiko-im/src/server/protocol.rs`（Rust側が正）と `Sources/KashikoIME/EngineProtocol.swift`（Swift側ミラー）
- キーイベントはエンジンが解釈する XKB keysym 表現に変換して送ります（`Sources/KashikoIME/KeyCodeMap.swift`）
- エンジンプロセスはクラッシュ時に指数バックオフで自動再起動し、スリープ復帰時にも再起動します（macOSがスリープ中にパイプを破棄するため）
- 候補ウィンドウはマウス操作に対応します。行の ✕ で候補を削除でき、下部のボタンで推論を追加実行できます。どちらもRustへのRPCとして送られ、更新された候補が非同期に返ります

## インストール

リポジトリ直下で実行します。

```bash
make install
```

その後、macOSからログアウトして再ログインし（新しいIMEを認識させるために必要）、`システム設定` > `キーボード` > `入力ソース` を編集 > `+` > 「日本語」から **Kashiko** を追加してください。

`make install` 時に設定中のモデル（メインと補助）を Hugging Face からダウンロードしてキャッシュに配置します。失敗した場合は初回変換時に自動でダウンロードされますが、完了まで候補が出るのに時間がかかります。

### 開発版の更新

ログアウトは不要です。

```bash
make install
```

次にテキストフィールドへフォーカスした時に自動で再起動します。

ただし**コード以外のメタデータ**（メニューアイコン、入力モード名、Info.plist の入力モード定義）は macOS 側にキャッシュされるため反映されないことがあります。その場合は `killall TextInputMenuAgent` を試し、それでも駄目なら入力ソースからKashikoを削除して再追加、最終手段としてログアウト・ログインしてください。

### テスト

```bash
make test
```

### リリース

`macos/Info.plist` の `CFBundleShortVersionString` を上げて `vX.Y.Z` タグを push すると、GitHub Actions（`.github/workflows/release.yml`）が `scripts/package-release.sh` でビルドしてリリースを作ります。添付されるのは、アプリの zip（Apple Silicon 用、ライセンス文書同梱）と `install.sh`、`SHA256SUMS` です。システム辞書は独立したリリース（Makefile の `DICT_RELEASE`、現在は `dict-20260226`）に置き、`install.sh` に SHA-256 を固定して埋め込みます。辞書を更新するときは新しいタグで `dict.tgz` を公開し、Makefile の `DICT_RELEASE` とハッシュを書き換えてください。

## システム辞書

モデル推論だけでは語彙が限られるため、システム辞書の併用を強く推奨します。`make install` 時に未インストールであればビルド済み辞書を自動でダウンロードして配置します。既に配置済みなら何もしないため、自前ビルドの辞書が上書きされることはありません。

辞書を自分でビルドする場合は [kashiko-tools の README](../crates/kashiko-tools/README.md) を参照してください。

## macOS 固有のキー

共通のキーバインドは [kashiko-im の README](../crates/kashiko-im/README.md#key-bindings) を参照してください。

| キー | 動作 |
|------|------|
| かな (JIS) | 日本語モードへ切替 |
| 英数 (JIS) | 変換中テキストを確定して直接入力モードへ切替 |
| 右⌘ 単独タップ | 日本語入力（ひらがな）へ戻る。直接入力モード、Shift+英字で入った英字モードのどちらからでも有効。⌘C など他のキーと組み合わせた場合は発動しない |

## 設定・データファイル

`directories` クレートの macOS 既定パス（`~/Library/Application Support/Kashiko/`）を使用します。

| ファイル | 内容 |
|---|---|
| `config.toml` | 設定 |
| `dict.bin` | システム辞書 |
| `user_dicts/` | ユーザー辞書 |
| `learning.tsv` | 学習キャッシュ（暗号化） |
| `learning.key` | 上記の暗号鍵（`0600`） |

## デバッグ

- ログ: `~/Library/Logs/Kashiko/kashiko-ime.log`（Swift側 NSLog と Rust側 tracing の両方）。入力内容が出るのは `debug`/`trace` レベルのみです
- サーバー単体デバッグ: `cargo run -p kashiko-im --bin kashiko-imserver` に JSON-RPC を直接流せます
- バンドルを組み立てずに開発中のサーバーを使う: `KASHIKO_IMSERVER=/path/to/kashiko-imserver`

## 既知の制約

ローマ字入力のみ対応です（かな入力レイアウトは未対応）。

## 参考プロジェクト

- [mac-akaza](https://github.com/akaza-im/mac-akaza)
