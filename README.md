# SAM Particle Counter

## Get Started

### Clone the repository

Clone the repository, then set up the virtual environment and install dependencies.

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

### Setup SAM2

Setup the Segment Anything Model2 (SAM2).

### Windows

```powershell
setup\setup_sam2.bat
```

### macOS/Linux

```bash
./setup/setup_sam2.sh
```

### Launch the application

```powershell
uv run main.py
```

## Troubleshooting / FAQ

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

