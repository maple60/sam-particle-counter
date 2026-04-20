# SAM Particle Counter

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
