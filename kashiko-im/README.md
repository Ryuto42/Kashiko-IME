# kashiko-im

macOS Swift フロントエンドから使われる日本語IMEエンジン。ローマ字→ひらがな変換、GPT-2ベースのニューラルかな漢字変換、学習キャッシュ、システム辞書を提供します。

フロントエンドのインストール手順:
- macOS: [kashiko-macos](../kashiko-macos/README.md)

## Features

- ニューラルかな漢字変換（llama.cppによるGGUF推論）
- 変換学習（ユーザーの変換履歴を記憶し、完全一致・前方一致で候補を優先表示）
- 日本語・英数字の混合入力（Shift切り替え）
- Surrounding Textによる文脈を考慮した変換
- システム辞書・ユーザー辞書による候補補完
- タイプミスの補正候補（キーボード配置を考慮した打鍵誤りの推定。変換応答後もバックグラウンドで探索を継続）
- ローマ字変換で崩れた英単語の復元（確度に応じて候補順位を決定）

> [!NOTE]
> モデル推論だけでは語彙が限られるため、システム辞書の併用を強く推奨します。システム辞書はIMEに同梱されていないため、別途インストールが必要です。詳しくは [Dictionary](#dictionary) を参照してください。

## Key Bindings

### ひらがな入力モード

| キー | 動作 |
|------|------|
| 文字キー | ローマ字入力 → ひらがな変換 |
| Space / Tab / ↓ | かな漢字変換を開始 |
| Enter | ひらがなのまま確定 |
| Escape | 入力をキャンセル |
| Backspace | 1文字削除 |
| Delete | カーソル位置の文字を削除 |
| ← → | カーソル移動 |
| Home / End | カーソルを先頭 / 末尾に移動 |
| Ctrl+K | カタカナモードに切り替え |
| Ctrl+Space | 全角スペースを入力 |

### 変換モード

| キー | 動作 |
|------|------|
| Space / Tab / ↓ | 次の候補 |
| ↑ | 前の候補 |
| ← → | 変換範囲を単語区切りで伸縮（残りは未変換のまま） |
| 1-9 | 候補を番号で選択・確定 |
| Enter | 選択中の候補を確定 |
| Escape | 変換をキャンセル（ひらがなに戻る） |
| 文字キー | 選択中の候補を確定して新しい入力を開始 |

確定した直後に `Ctrl+Backspace` を押すと、確定を取り消して変換モードに戻せます（標準のCocoaテキスト部品で動作。一部のElectron系アプリでは効かない場合があります）。

### モード切り替え

| キー | 動作 |
|------|------|
| Shift+英字 | 英数字モードに切り替え + 大文字入力 |
| Ctrl+K | カタカナモードに切り替え |
| Right Super | 英数字/カタカナ → ひらがなモードに復帰 |
| Ctrl+Shift+L | ライブ変換のON/OFF |

### 英数字モード

英数字モードでは文字がローマ字変換されず、そのまま入力されます。日本語と英語を混ぜて入力し、Spaceで変換するとひらがな部分のみ変換されます。

例: `わたしはLinuxが` → 変換 → `私はLinuxが`

## Configuration

設定ファイル: `~/Library/Application Support/com.kashiko.kashiko-im/config.toml`

```toml
[conversion]
live_conversion = true          # ライブ変換を起動時に有効化（Ctrl+Shift+L で実行中も切替。既定ON）
composing_chunk_len = 30        # ライブ変換で1回のモデル変換が扱う読みの最大文字数（= 1キーあたりレイテンシの上限）
strategy = "adaptive"           # 変換ストラテジー（adaptive / light / main）
num_candidates = 9              # 変換候補数（Space押下時）
n_threads = 4                   # 推論スレッド数（0 = 全コア使用）
model = "jinen-v1-small-q5"     # メインモデル（モデルID or GGUFパス）
light_model = "jinen-v1-xsmall-q5"  # 軽量モデル（ビームサーチ・長文用）
use_context = true              # Surrounding Textを変換に使用する
max_context_length = 10         # コンテキストの最大文字数
short_input_threshold = 10      # ビームサーチを使うトークン数の上限
beam_width = 3                  # ビーム幅
max_latency_ms = 100            # メインモデルの許容レイテンシ（ms）。超過時は軽量モデルに自動切替（0 = 無効）
dict_path = "/path/to/dict.bin" # システム辞書パス（省略時はデータディレクトリの dict.bin。[Dictionary](#dictionary) 参照）

[learning]
enabled = true                 # 変換学習の有効/無効
max_entries = 10000            # 学習エントリの最大数
```

> [!NOTE]
> 上記は主要な設定項目の抜粋です。全項目の正確な既定値と説明は [`config/default.toml`](config/default.toml) を参照してください（各設定行に日本語コメント付き）。

### Live Conversion

入力と同時にかな漢字変換の結果をプリエディットへリアルタイム表示します（Spaceを押さずに変換が進む）。`Ctrl+Shift+L` でON/OFFを切り替えられ、既定では `live_conversion = true` で有効です。

長文入力でも1キーあたりのレイテンシを一定に保つため、変換中のバッファを内部で最大 `composing_chunk_len` 文字（既定30）のチャンクに分割し、編集した箇所のチャンクだけを再変換します。チャンクは内部的な分割で、ユーザーには連続した1つのプリエディットとして見えます。記号・数字の連続は日本語とは別チャンクに分けてそのまま通すため、`123456` のような並びが変換で崩れることはありません。

### Conversion Strategy

`strategy` で変換時のモデル使い分けを制御できます。

| 値 | 説明 | 読み込むモデル |
|---|---|---|
| `adaptive` | デフォルト。レイテンシに応じてメイン・軽量モデルを動的に切り替え | メイン + 軽量 |
| `light` | 軽量モデルのみ使用。メモリ消費が少なく、低スペックPCにおすすめ | 軽量のみ |
| `main` | メインモデルのみ使用（ビームサーチなし） | メインのみ |

低スペックのPC（メモリが少ない、CPUが遅い等）では `strategy = "light"` を設定すると、軽量モデル1つだけで動作するためメモリ使用量が削減され、レスポンスも安定します。

```toml
[conversion]
strategy = "light"
```

### Performance Tuning

CPU高負荷時（Rustビルド中など）にかな漢字変換が遅くなる場合は、`n_threads` を小さくするとレスポンスが改善します。

### Dictionary

辞書の構築・管理については [kashiko-cli の README](../kashiko-cli/README.md) を参照してください。

#### System Dictionary

double-array trieベースのシステム辞書で、モデル推論に加えて辞書からの変換候補を提供します。

- デフォルトパス: `~/Library/Application Support/com.kashiko.kashiko-im/dict.bin`
- `dict_path` で任意のパスを指定可能
- ファイルが存在しない場合は辞書なしで動作

`make install` が自動でダウンロードしますが、手動で配置する場合は以下のとおりです:

```bash
curl -LO https://github.com/togatoga/kashiko/releases/download/v0.1.0/dict.tgz
tar xzf dict.tgz
mkdir -p ~/Library/"Application Support"/com.kashiko.kashiko-im
cp dict.bin ~/Library/"Application Support"/com.kashiko.kashiko-im/
```

自分でビルドする場合は [kashiko-cli の README](../kashiko-cli/README.md) を参照してください。

#### User Dictionary

ユーザー辞書ディレクトリにファイルを配置すると、ユーザー辞書として読み込まれます。

- デフォルトパス: `~/Library/Application Support/com.kashiko.kashiko-im/user_dicts/`
- ディレクトリ内のファイルはすべて自動で読み込み（KRKNバイナリ・Mozc TSV を自動判定）
- ディレクトリが存在しない場合はユーザー辞書なしで動作

変換候補の優先順位:

1. 📝 学習キャッシュ
2. 👤 ユーザー辞書
3. 🤖 モデル推論
4. 📚 システム辞書（スコア順）
5. 🔄 Rewriter（英字全角半角・記号バリアント）
6. ひらがな / カタカナ（常に最下部）

半角カタカナは候補に表示しません（全角形と並ぶとノイズになるため）。

### Learning Cache

ユーザーが選択した変換結果を記憶し、次回以降の変換で優先表示します。

- 保存先: `~/Library/Application Support/com.kashiko.kashiko-im/learning.tsv`
- 完全一致と前方一致（予測変換）の両方に対応
  - 例: 「早稲田大学」を一度変換すると、次回「わせだ」と入力した時点で候補に表示
- 学習候補は変換時（Space押下後）に最大3件表示。入力中は候補ウィンドウを出さず、ライブ変換のプリエディットのみを表示します
- スコアはrecency（最終使用日時）重視 + 頻度補正
- IME切り替え・ウィンドウ切り替え時に自動保存（commit のたびには保存しない）
- `[learning] enabled = false` で無効化可能
- 学習履歴を削除するには: `rm ~/Library/"Application Support"/com.kashiko.kashiko-im/learning.tsv`
