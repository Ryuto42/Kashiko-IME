# kashiko-tools

kashiko-engine を利用した開発用のコマンドラインツール群。**IMEの動作には不要です。**

## Binaries

| バイナリ | 概要 |
|---|---|
| `kashiko-dict` | 辞書のビルド（`build`：JSON / Mozc TSV → バイナリ）と検索（`search`） |
| `sudachi-dict` | Sudachi CSV から JSON 辞書を生成 |
| `ajimee-bench` | AJIMEE-Bench による変換精度と遅延（p50/p90）の評価（`--gguf`/`--tokenizer-json`/`--pre-tokenizer`/`--prompt` で他のモデルも比較可能。`--threads 2` で IME と同じスレッド数） |

各コマンドのオプションは `--help` で確認できます。

## Build

```bash
cargo build -p kashiko-tools --release
```

## 辞書の構築

IMEが使うシステム辞書は double-array trie のバイナリです。ソースからビルドする流れは次のとおりです。

1. [SudachiDict](http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict-raw/) の辞書CSVを入手する
2. `sudachi-dict` で JSON 辞書に変換する。既定では Sudachi の正規コストをそのまま使い、`--model-scores` を指定すると jinen モデルの NLL で候補を順序付けします
3. `kashiko-dict build` でバイナリ辞書に変換し、`~/Library/Application Support/Kashiko/dict.bin` に配置する（ライセンス表記は [docs/dictionary](../../docs/dictionary/) にあります）

ビルド済み辞書は `make install` が自動でダウンロードするため、通常この手順は不要です。

## License

MIT OR Apache-2.0
