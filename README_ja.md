# SAM Particle Counter

SAM Particle Counter は、SAM2 を使ったセグメンテーションと Napari ベースの可視化により、画像データから粒子をカウントするデスクトップワークフローです。

## 前提条件

セットアップ前に以下の要件を確認してください。

- **Python**: `>=3.11.6`（`pyproject.toml` で定義）
- **必須ツール**: 依存関係・仮想環境管理に [`uv`](https://docs.astral.sh/uv/) を使用
- **GPU/CUDA**:
  - GPU は**任意**です。CPU でも動作しますが、パフォーマンスが低下する場合があります。
  - GPU アクセラレーションを使用する場合は、CUDA 対応の NVIDIA GPU と、環境に合った CUDA 対応 PyTorch ビルドが必要です。
  - セットアップ実行前に、OS/ドライバと PyTorch の CUDA 互換性を確認してください。
- **OS 別の注意事項**:
  - **Windows**: 以下の PowerShell コマンドを使用し、`.bat` セットアップスクリプトを実行してください。
  - **macOS**: 以下のシェルコマンドを使用し、`.sh` セットアップスクリプトを実行してください。
  - **Linux**: 以下のシェルコマンドを使用し、`.sh` セットアップスクリプトを実行してください。

### メンテナンスタスク（メタデータの一貫性）

- `pyproject.toml` の `[project].description` を README のプロジェクト説明と一致させてください（`Add your description here` などのプレースホルダーは残さないこと）。

## はじめに

### リポジトリを準備する

リポジトリをクローン後、仮想環境をセットアップして依存関係をインストールしてください。

#### Windows

```powershell
uv sync
.venv\Scripts\activate
```

#### macOS/Linux

```bash
uv sync
source .venv/bin/activate
```

### SAM2 セットアップ

Segment Anything Model2（SAM2）をセットアップします。

### Windows

```powershell
setup\setup_sam2.bat
```

### macOS/Linux

```bash
./setup/setup_sam2.sh
```

### アプリ起動

```powershell
uv run main.py
```

## Quick Start Workflow

Napari上で粒子カウントを行う際の典型的な操作フローです。`main.py` のUIボタン名に合わせて記載しているため、READMEと画面を1対1で対応付けできます。

1. **画像を読み込む**  
   Napariで画像を開きます（ドラッグ&ドロップ / `File > Open...` など）。最初の画像追加時に、対応するROIレイヤー（`<画像名>_ROI`）が自動作成されます。
2. **ROIレイヤー作成と矩形描画**  
   `*_ROI` レイヤーを選択し、Shapesツールで矩形ROIを1つ描画します（複数ある場合は最後に描いたROIが使用されます）。
3. **Crop実行**  
   右側ドックの **`Crop to ROI`** を実行して、`<画像名>_cropped` レイヤーを作成します。
4. **SAM2 auto segmentation実行**  
   `*_cropped`（または対応レイヤー）を選び、右側ドックの **`Run SAM2 auto segmentation`** を実行します。必要に応じて `Output mode` などのパラメータを調整してください。
5. **粒子数確認・エクスポート**  
   セグメンテーション結果を確認後、右側ドックの **`Export segmentation artifacts`** を実行して書き出します。完了メッセージ内の `sam2=...` / `final=...` が粒子数の目安です。

## トラブルシューティング / FAQ

### 1) `setup/setup_sam2.sh` / `.bat` 実行時の失敗例と対処

| よくある症状 | 主な原因 | 対処 |
|---|---|---|
| `[Error] uv is not installed or not on PATH.` | `uv` が未インストール、または PATH 未設定 | `uv --version` で確認し、未導入ならインストール。新しいシェルで再実行。 |
| `[Error] git is not installed or not on PATH.` | `git` が未インストール、または PATH 未設定 | `git --version` を確認。未導入なら Git を導入。 |
| `Neither curl nor wget was found.`（sh） / `Neither curl, pwsh, nor powershell was found.`（bat） | ダウンローダー不在 | macOS/Linux は `curl` または `wget`、Windows は `curl`/`pwsh`/`powershell` を利用可能にする。 |
| `uv sync failed.` / `uv sync` で停止 | 依存解決失敗、ネットワーク、Python バージョン不一致 | 下記「2) `uv sync` 失敗時の確認項目」を順に確認。 |
| `Failed to clone SAM2 repository.` / fetch/pull/check out に失敗 | ネットワーク制限、GitHub アクセス不可、既存ディレクトリ破損 | `sam2_repo` の状態を確認し、必要なら一度退避・削除して再実行。プロキシ環境では Git 設定も確認。 |
| `Failed to download ...` | モデルチェックポイント取得時の通信失敗 | 通信状態を確認後、再実行。途中生成された壊れた `.pt` はスクリプトが削除する想定。 |
| `Failed to install SAM2.` / `uv pip install -e .` で失敗 | CUDA ビルド要件不足、ビルド環境不足 | まず `SKIP_SAM2_CUDA=1` に変更して再実行し、CPU ベースで導入確認。 |

---

### 2) `uv sync` 失敗時の確認項目

1. **Python バージョン**: 本プロジェクトは `requires-python = ">=3.11.6"` を要求します。`python --version` を確認してください。  
2. **`uv` のバージョン / PATH**: `uv --version` が通ること。  
3. **仮想環境の再作成**: `.venv` が壊れている場合は削除して `uv sync` を再実行。  
4. **ネットワーク / プロキシ**: 社内ネットワークや証明書設定で PyPI / GitHub へのアクセスが遮断されていないか確認。  
5. **ロック・キャッシュ由来の不整合**: `uv cache clean` 実行後に再度 `uv sync` を試す。  
6. **OS 権限**: セキュリティソフトや権限不足で `.venv` やキャッシュへの書き込みが阻害されていないか確認。  

---

### 3) レイヤー名不一致・ROI未作成時の典型エラー（`main.py` の警告対応）

| よくある症状（警告メッセージ） | 主な原因 | 対処 |
|---|---|---|
| `ROIが1つもありません。矩形を1つ描いてください。` | ROI レイヤーに矩形が未作成 | `<画像名>_ROI` を選択し、矩形を1つ以上描いてから `Crop to ROI` を実行。 |
| `有効なROIが描かれていません。` | ROI が画像範囲外、または面積ゼロ | ROI を描き直し、画像内に十分な面積を持つ矩形にする。 |
| `<画像名> という画像レイヤーが見つかりません。` | `<画像名>_ROI` に対応する元画像レイヤー名が不一致 | 画像レイヤー名と ROI レイヤー名の接頭辞を一致させる（例: `sample` と `sample_ROI`）。 |
| `<レイヤー名> は画像レイヤーではありません。` | 同名の別タイプレイヤーを参照している | 対応レイヤーが Image 型であることを確認し、名前重複を解消。 |
| `<...>_sam2_auto_labels ... が見つかりません。先に cropped 画像に対して SAM2 auto segmentation を実行してください。` | SAM2 auto segmentation 未実行、または命名不一致 | 先に cropped 画像で SAM2 auto segmentation を実行し、生成レイヤー名を確認。 |
| `<レイヤー名> は Labels レイヤーではありません。` | `*_sam2_auto_labels` が Labels 型でない | 既存レイヤーを削除して再生成、または正しい Labels レイヤーを選択。 |

---

### 4) 動作確認済み環境（OS / Python）

以下は **現時点で README 上に明示されているサポート対象** です。実運用では GPU / ドライバ / ネットワーク条件により結果が変わる場合があります。

| OS | Python | 補足 |
|---|---|---|
| Windows | 3.11.6 以上 | `setup/setup_sam2.bat` を利用 |
| macOS | 3.11.6 以上 | `setup/setup_sam2.sh` を利用 |
| Linux | 3.11.6 以上 | `setup/setup_sam2.sh` を利用 |
