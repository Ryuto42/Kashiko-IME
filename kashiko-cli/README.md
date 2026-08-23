# kashiko-cli

kashiko-engine を利用した開発用CLIツール群。**IMEの動作には不要です。**

## Binaries

| バイナリ | 概要 |
|---|---|
| `kashiko-dict` | 辞書のビルド（JSON / Mozc TSV → バイナリ）とビューア（Web UI + CLI検索） |
| `sudachi-dict` | Sudachi CSV から JSON 辞書を生成 |
| `kashiko-server` | かな漢字変換 HTTP サーバー（Web UI 付き） |
| `ajimee-bench` | AJIMEE-Bench による変換精度の評価 |

各コマンドのオプションは `--help` で確認できます。

## Build

```bash
cargo build -p kashiko-cli --release
```

## 辞書の構築

IMEが使うシステム辞書は double-array trie のバイナリです。ソースからビルドする流れは次のとおりです。

1. [SudachiDict](http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict-raw/) の辞書CSVを入手する
2. `sudachi-dict` で JSON 辞書に変換する。既定では Sudachi の正規コストをそのまま使い、`--model-scores` を指定すると jinen モデルの NLL で候補を順序付けします
3. `kashiko-dict build` でバイナリ辞書に変換し、`~/Library/Application Support/com.kashiko.kashiko-im/dict.bin` に配置する

ビルド済み辞書は `make install` が自動でダウンロードするため、通常この手順は不要です。

## サーバー・ビューア

`kashiko-server` と `kashiko-dict view` はローカルの HTTP サーバーを起動します。既定のバインドアドレスは `127.0.0.1` で、外部からは接続できません。

## License

MIT OR Apache-2.0
