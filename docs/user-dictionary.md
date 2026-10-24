# ユーザー辞書

kashiko IME はユーザー辞書をサポートしています。ユーザーが独自の単語を登録し、変換候補に反映させることができます。

## 対応形式

### Mozc/Google IME TSV 形式

タブ区切りのテキストファイルです。Google 日本語入力やMozcからエクスポートした辞書、Microsoft IME の「テキストファイルで出力」したファイル（UTF-16）をそのまま使用できます。

```
# コメント行（#で始まる行は無視されます）
ヨミ	表層形	品詞	コメント
```

例:
```
かしこ	Kashiko	固有名詞	IME名
にこにこ	ニコニコ	名詞	動画サイト
```

### バイナリ形式

`kashiko-dict build` コマンドでビルドしたバイナリ辞書です。読み込みが高速です。

**巨大な辞書（数万エントリ以上）は必ずバイナリ形式に変換してください。**
Mozc TSV 形式のまま読み込むと、IME起動時にダブル配列の構築が走るため起動が遅くなります。
バイナリ形式ならダブル配列が構築済みなので瞬時に読み込めます。

```bash
# Mozc TSV から バイナリを作成（リポジトリ直下で実行）
cargo run --release --bin kashiko-dict -- build --format mozc input.tsv -o user_dict.bin
```

## 設定方法

ユーザー辞書ディレクトリに辞書ファイルを配置するだけで自動的に読み込まれます。

```
~/Library/Application Support/Kashiko/user_dicts/
├── my_dict.txt          ← Mozc TSV 形式
├── nico_dict.bin        ← KRKN バイナリ形式
└── another.txt          ← ファイルを置くだけで有効
```

- 初回起動時にディレクトリが自動作成されます
- ディレクトリ内の全ファイルが自動読み込みされます（設定不要）
- Mozc TSV と バイナリを混在可能（ファイル先頭で自動判別）
- ファイル名のアルファベット順に読み込まれ、先のファイルが優先されます
- ディレクトリが存在しない場合はユーザー辞書なしで動作します

## 候補の優先順位

変換時の候補は以下の優先順位で表示されます:

1. **👤 ユーザー** — ユーザー辞書の候補
2. **🤖 AI** — AI モデルの推論結果
3. **📚 辞書** — システム辞書の候補
4. ひらがな / カタカナ（フォールバック）

## 外部辞書の利用例

ニコニコ大百科・Pixiv辞書（Google IME形式）:

```bash
# ディレクトリ作成
mkdir -p ~/Library/"Application Support"/Kashiko/user_dicts

# ダウンロードして配置
curl -L -o "$HOME/Library/Application Support/Kashiko/user_dicts/nico_dict.txt" \
  https://raw.githubusercontent.com/ncaq/dic-nico-intersection-pixiv/master/public/dic-nico-intersection-pixiv-google.txt
```

## 辞書の確認 (kashiko-dict search)

`kashiko-dict search` コマンドで辞書の内容を検索・確認できます。リポジトリ直下で実行します。

```bash
# ヨミで検索（完全一致）
cargo run --release --bin kashiko-dict -- search -q きょう user_dict.txt

# ヨミで前方一致検索
cargo run --release --bin kashiko-dict -- search -q きょう -p user_dict.txt

# 表層形で検索（部分一致）
cargo run --release --bin kashiko-dict -- search -q 京都 -s user_dict.txt

# 全エントリをダンプ
cargo run --release --bin kashiko-dict -- search -a user_dict.txt

# 複数の辞書をまとめて検索
cargo run --release --bin kashiko-dict -- search -q きょう dict1.bin dict2.txt
```

## 辞書ビルダー (kashiko-dict build)

テキスト辞書を高速なバイナリ形式に変換します:

```bash
# JSON 形式から（デフォルト）
cargo run --release --bin kashiko-dict -- build input.json -o dict.bin

# Mozc TSV 形式から
cargo run --release --bin kashiko-dict -- build --format mozc input.tsv -o dict.bin

# 形式は拡張子で自動判別（.json → JSON、それ以外 → Mozc TSV）
cargo run --release --bin kashiko-dict -- build input.txt -o dict.bin
```
