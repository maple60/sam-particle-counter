from napari.utils.notifications import show_warning, show_info
import napari
from napari.layers import Image, Points, Labels
from magicgui import magicgui
import numpy as np
import cv2
import csv
import json
import platform
import random
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from napari.utils.events import Event
from napari import Viewer
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox


class RoiController:
    def __init__(self, viewer: Viewer):
        self.viewer = viewer
        self.viewer.layers.events.inserted.connect(self.on_image_layer_added)
        self._is_first_image_initialized = False
        self._sam2_mask_generators: dict[
            tuple[int, float, float, int, int], object
        ] = {}
        self._sam2_image_predictor = None
        self._register_widgets()

    @dataclass(slots=True)
    class Sam2AutoParams:
        points_per_side: int
        pred_iou_thresh: float
        stability_score_thresh: float
        crop_n_layers: int
        crop_n_points_downscale_factor: int
        deduplicate_masks: bool
        contain_thresh: float
        iou_thresh: float
        score_key: str
        output_mode: str

    @dataclass(slots=True)
    class LayerNameSet:
        image: str
        roi: str
        cropped: str
        prompts_fg: str
        prompts_bg: str
        prompts_legacy: str
        sam2_auto_shapes: str
        sam2_auto_labels: str
        sam2_auto_labels_original: str
        sam2_points: str

    def _layer_names(self, image_layer_name: str) -> LayerNameSet:
        return self.LayerNameSet(
            image=image_layer_name,
            roi=f"{image_layer_name}_ROI",
            cropped=f"{image_layer_name}_cropped",
            prompts_fg=f"{image_layer_name}_prompts_fg",
            prompts_bg=f"{image_layer_name}_prompts_bg",
            prompts_legacy=f"{image_layer_name}_prompts",
            sam2_auto_shapes=f"{image_layer_name}_sam2_auto",
            sam2_auto_labels=f"{image_layer_name}_sam2_auto_labels",
            sam2_auto_labels_original=f"{image_layer_name}_sam2_auto_labels_original",
            sam2_points=f"{image_layer_name}_sam2_points",
        )

    def _infer_image_name_from_layer(self, layer) -> str | None:
        if isinstance(layer, Image):
            if layer.name.endswith("_cropped"):
                return layer.name[: -len("_cropped")]
            return layer.name

        if not hasattr(layer, "name"):
            return None

        layer_name = str(layer.name)
        suffixes = (
            "_ROI",
            "_prompts_fg",
            "_prompts_bg",
            "_prompts",
            "_sam2_auto",
            "_sam2_auto_labels",
            "_sam2_auto_labels_original",
            "_sam2_points",
        )
        for suffix in suffixes:
            if layer_name.endswith(suffix):
                return layer_name[: -len(suffix)]
        return None

    def _resolve_target_cropped_image_layer(self) -> Image | None:
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            return None

        if isinstance(active_layer, Image) and active_layer.name.endswith("_cropped"):
            return active_layer

        image_name = self._infer_image_name_from_layer(active_layer)
        if not image_name:
            return None
        if image_name.endswith("_cropped"):
            image_name = image_name[: -len("_cropped")]

        cropped_name = self._layer_names(image_name).cropped
        if cropped_name not in self.viewer.layers:
            return None

        cropped_layer = self.viewer.layers[cropped_name]
        if not isinstance(cropped_layer, Image):
            return None

        return cropped_layer

    def _resolve_sam2_auto_labels_layer(self, cropped_layer: Image) -> Labels | None:
        cropped_name = cropped_layer.name
        base_name = (
            cropped_name[: -len("_cropped")]
            if cropped_name.endswith("_cropped")
            else cropped_name
        )
        candidate_names = [
            self._layer_names(cropped_name).sam2_auto_labels,
            self._layer_names(base_name).sam2_auto_labels,
        ]

        for layer_name in candidate_names:
            if layer_name not in self.viewer.layers:
                continue

            layer = self.viewer.layers[layer_name]
            if not isinstance(layer, Labels):
                show_warning(f"{layer_name} は Labels レイヤーではありません。")
                return None

            if layer.data.shape[:2] != cropped_layer.data.shape[:2]:
                continue

            return layer

        show_warning(
            (
                f"{candidate_names[0]} または {candidate_names[1]} が見つかりません。"
                "先に cropped 画像に対して SAM2 auto segmentation を実行してください。"
            )
        )
        return None

    def _resolve_sam2_original_labels_layer(
        self, cropped_layer: Image
    ) -> Labels | None:
        cropped_name = cropped_layer.name
        base_name = (
            cropped_name[: -len("_cropped")]
            if cropped_name.endswith("_cropped")
            else cropped_name
        )
        candidate_names = [
            self._layer_names(cropped_name).sam2_auto_labels_original,
            self._layer_names(base_name).sam2_auto_labels_original,
        ]

        for layer_name in candidate_names:
            if layer_name not in self.viewer.layers:
                continue
            layer = self.viewer.layers[layer_name]
            if not isinstance(layer, Labels):
                continue
            if layer.data.shape[:2] != cropped_layer.data.shape[:2]:
                continue
            return layer
        return None

    def _register_widgets(self) -> None:
        """
        @magicgui(call_button="選択中レイヤーを解析")
        def analyze_active() -> None:
            layer = self.viewer.layers.selection.active

            if layer is None or not isinstance(layer, Image):
                print("画像レイヤーを1つだけ選択してください。")
                return

            arr = np.asarray(layer.data)
            print(layer.name, arr.shape, arr.dtype)

        self.analyze_active_widget = analyze_active
        """

        @magicgui(call_button="Crop to ROI")
        def crop_to_roi() -> None:
            layer = self.viewer.layers.selection.active
            # ROIレイヤーの場合のみ処理を続行
            if not layer.name.endswith("_ROI"):
                show_info("ROIレイヤーを選択してください。")
                return

            # 末尾の_ROIを除いた名前が画像レイヤ
            image_layer_name = layer.name[:-4]
            layer_names = self._layer_names(image_layer_name)
            try:
                image_layer = self.viewer.layers[layer_names.image]
            except KeyError:
                show_warning(f"{image_layer_name} という画像レイヤーが見つかりません。")
                return
            if not isinstance(image_layer, Image):
                show_warning(f"{image_layer_name} は画像レイヤーではありません。")
                return
            show_info("対応する画像レイヤーを取得できました。")  # デバッグ用

            # ROIが1つもない場合は処理を続行しない
            if len(layer.data) == 0:
                show_warning("ROIが1つもありません。矩形を1つ描いてください。")
                return
            # 最後に描いたROIを取得
            roi = layer.data[-1]
            show_info(f"ROIの頂点座標: {roi}")  # デバッグ用
            # rectangleの4頂点からbboxを取得
            ymin = int(np.floor(roi[:, 0].min()))
            ymax = int(np.ceil(roi[:, 0].max()))
            xmin = int(np.floor(roi[:, 1].min()))
            xmax = int(np.ceil(roi[:, 1].max()))
            # 画像の範囲外にならないように補正
            h, w = image_layer.data.shape[:2]
            ymin = max(0, ymin)
            ymax = min(h, ymax)
            xmin = max(0, xmin)
            xmax = min(w, xmax)

            if ymin >= ymax or xmin >= xmax:
                show_warning("有効なROIが描かれていません。")
                return

            cropped_layer = image_layer.data[ymin:ymax, xmin:xmax]
            cropped_layer = self.viewer.add_image(
                cropped_layer, name=layer_names.cropped
            )
            self._attach_metadata(
                cropped_layer,
                {
                    "source_image": layer_names.image,
                    "source_roi_layer": layer_names.roi,
                    "crop_bbox_yx": [ymin, ymax, xmin, xmax],
                    "layer_names": asdict(layer_names),
                },
            )
            show_info(f"cropped: y={ymin}:{ymax}, x={xmin}:{xmax}")  # デバッグ用

            # 新しいcropped画像をアクティブにする
            self.viewer.layers.selection.select_only(cropped_layer)
            # cropped画像の中心へカメラを移動
            self._zoom_to_layer(cropped_layer)

        self.crop_to_roi_widget = crop_to_roi

        @magicgui(
            call_button="Run SAM2 auto segmentation",
            points_per_side={"min": 8, "max": 128, "step": 1},
            pred_iou_thresh={"min": 0.0, "max": 1.0, "step": 0.01},
            stability_score_thresh={"min": 0.0, "max": 1.0, "step": 0.01},
            crop_n_layers={"min": 0, "max": 3, "step": 1},
            crop_n_points_downscale_factor={"min": 1, "max": 4, "step": 1},
            contain_thresh={"min": 0.0, "max": 1.0, "step": 0.01},
            iou_thresh={"min": 0.0, "max": 1.0, "step": 0.01},
            output_mode={"choices": ["labels", "shapes", "both"]},
        )
        def run_sam2_auto_on_active(
            points_per_side: int = 56,
            pred_iou_thresh: float = 0.84,
            stability_score_thresh: float = 0.90,
            crop_n_layers: int = 1,
            crop_n_points_downscale_factor: int = 1,
            deduplicate_masks: bool = True,
            contain_thresh: float = 0.90,
            iou_thresh: float = 0.70,
            score_key: str = "predicted_iou",
            output_mode: str = "labels",
        ) -> None:
            layer = self.viewer.layers.selection.active

            if layer is None or not isinstance(layer, Image):
                show_warning("画像レイヤーを1つだけ選択してください。")
                return

            try:
                mask_generator = self._get_sam2_mask_generator(
                    points_per_side=points_per_side,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    crop_n_layers=crop_n_layers,
                    crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                )
                image_rgb = self._prepare_rgb_image(layer)
                masks = mask_generator.generate(image_rgb)
            except Exception as e:
                show_warning(f"SAM2自動セグメンテーションに失敗しました: {e}")
                traceback.print_exc()
                return

            masks = self._sort_masks_by_position(masks, method="centroid")
            removed_pairs: list[tuple[int, int, dict[str, float]]] = []
            if deduplicate_masks:
                masks, removed_pairs = self._deduplicate_masks(
                    masks,
                    contain_thresh=contain_thresh,
                    iou_thresh=iou_thresh,
                    score_key=score_key,
                )

            needs_shapes = output_mode in ("shapes", "both")
            needs_labels = output_mode in ("labels", "both")
            layer_names = self._layer_names(layer.name)
            run_params = self.Sam2AutoParams(
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                deduplicate_masks=deduplicate_masks,
                contain_thresh=contain_thresh,
                iou_thresh=iou_thresh,
                score_key=score_key,
                output_mode=output_mode,
            )
            run_metadata = {
                "mode": "sam2_auto",
                "image_layer": layer.name,
                "layer_names": asdict(layer_names),
                "sam2_params": asdict(run_params),
                "mask_count_after_processing": len(masks),
                "dedup_removed_count": len(removed_pairs),
            }

            polygons: list[np.ndarray] = []
            polygon_scores: list[float] = []
            if needs_shapes:
                polygons, polygon_scores = self._masks_to_polygons(masks)
                if not polygons:
                    show_info("SAM2で有効なマスクが得られませんでした。")
                    return

            label_image: np.ndarray | None = None
            if needs_labels:
                label_image = self._masks_to_label_image(masks, layer.data.shape[:2])
                if np.max(label_image) == 0:
                    show_info("SAM2で有効なマスクが得られませんでした。")
                    return

            if needs_shapes:
                shapes_name = layer_names.sam2_auto_shapes
                if any(existing.name == shapes_name for existing in self.viewer.layers):
                    del self.viewer.layers[shapes_name]

                shapes_layer = self.viewer.add_shapes(
                    polygons,
                    shape_type="polygon",
                    name=shapes_name,
                    properties={"score": np.asarray(polygon_scores)},
                    edge_color="#5E409D",  # purple-600
                    face_color="#5E409D",
                    edge_width=1,
                    opacity=0.3,
                )
                self._attach_metadata(
                    shapes_layer,
                    {
                        **run_metadata,
                        "output_layer": shapes_name,
                        "output_type": "shapes",
                        "polygon_count": len(polygons),
                    },
                )

            if needs_labels and label_image is not None:
                labels_name = layer_names.sam2_auto_labels
                if any(existing.name == labels_name for existing in self.viewer.layers):
                    del self.viewer.layers[labels_name]
                original_labels_name = layer_names.sam2_auto_labels_original
                if any(
                    existing.name == original_labels_name
                    for existing in self.viewer.layers
                ):
                    del self.viewer.layers[original_labels_name]

                labels_layer = self.viewer.add_labels(
                    label_image,
                    name=labels_name,
                    opacity=0.7,
                )
                original_labels_layer = self.viewer.add_labels(
                    label_image.copy(),
                    name=original_labels_name,
                    opacity=0.0,
                    visible=False,
                )
                original_labels_layer.editable = False
                # set seleceted_label to max label so that the last mask is selected by default
                labels_layer.selected_label = int(np.max(label_image))
                self._attach_metadata(
                    labels_layer,
                    {
                        **run_metadata,
                        "output_layer": labels_name,
                        "output_type": "labels",
                        "label_count": int(np.max(label_image)),
                    },
                )
                self._attach_metadata(
                    original_labels_layer,
                    {
                        **run_metadata,
                        "output_layer": original_labels_name,
                        "output_type": "labels_original",
                        "label_count": int(np.max(label_image)),
                    },
                )
                # Keep editable labels layer active after creating hidden original layer
                self.viewer.layers.selection.select_only(labels_layer)

            if deduplicate_masks:
                show_info(
                    f"SAM2 auto masks: {len(masks)} (dedup removed={len(removed_pairs)}, output={output_mode})"
                )
            else:
                show_info(f"SAM2 auto masks: {len(masks)} (output={output_mode})")

        self.run_sam2_auto_on_active_widget = run_sam2_auto_on_active

        @magicgui(
            call_button="Create prompt points layer",
            auto_call=True,
            prompt_target={
                "choices": [
                    "Foreground points",
                    "Background points",
                ]
            },
        )
        def create_prompt_points_layer(
            prompt_target: str = "Foreground points",
        ) -> None:
            layer = self._resolve_target_cropped_image_layer()
            if layer is None:
                show_warning(
                    "対象の cropped 画像レイヤーが見つかりません。先に Crop to ROI を実行してください。"
                )
                return

            fg_points_name = f"{layer.name}_prompts_fg"
            bg_points_name = f"{layer.name}_prompts_bg"
            layer_names = self._layer_names(layer.name)

            if bg_points_name in self.viewer.layers:
                bg_layer = self.viewer.layers[bg_points_name]
                if not isinstance(bg_layer, Points):
                    show_warning(f"{bg_points_name} はPointsレイヤーではありません。")
                    return
            else:
                bg_layer = self.viewer.add_points(
                    data=np.empty((0, 2), dtype=np.float32),
                    name=layer_names.prompts_bg,
                    features={"label": np.empty((0,), dtype=np.int32)},
                    face_color="#EF4444",
                    border_color="#7F1D1D",
                    size=14,
                )
                self._attach_metadata(
                    bg_layer,
                    {
                        "mode": "sam2_prompt_points",
                        "prompt_label": 0,
                        "image_layer": layer.name,
                        "layer_names": asdict(layer_names),
                    },
                )

            if fg_points_name in self.viewer.layers:
                fg_layer = self.viewer.layers[fg_points_name]
                if not isinstance(fg_layer, Points):
                    show_warning(f"{fg_points_name} はPointsレイヤーではありません。")
                    return
            else:
                fg_layer = self.viewer.add_points(
                    data=np.empty((0, 2), dtype=np.float32),
                    name=layer_names.prompts_fg,
                    features={"label": np.empty((0,), dtype=np.int32)},
                    face_color="#22C55E",
                    border_color="#166534",
                    size=14,
                )
                self._attach_metadata(
                    fg_layer,
                    {
                        "mode": "sam2_prompt_points",
                        "prompt_label": 1,
                        "image_layer": layer.name,
                        "layer_names": asdict(layer_names),
                    },
                )

            target_layer = (
                fg_layer if prompt_target == "Foreground points" else bg_layer
            )
            target_layer.mode = "add"
            self.viewer.layers.selection.select_only(target_layer)
            show_info(
                "ポイントレイヤー準備OK。プルダウンで選んだ対象に点を追加してください（前景=緑, 背景=赤）。追加後に Run SAM2 point prompt を実行します。"
            )

        self.create_prompt_points_layer_widget = create_prompt_points_layer
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed_for_prompt_target
        )
        self._sync_prompt_target_with_active_layer()

        @magicgui(call_button="Run SAM2 point prompt")
        def run_sam2_point_prompt() -> None:
            image_layer = self._resolve_target_cropped_image_layer()
            if image_layer is None:
                show_warning(
                    "対象の cropped 画像レイヤーが見つかりません。先に Crop to ROI を実行してください。"
                )
                return

            try:
                point_coords, point_labels = self._collect_point_prompts(image_layer)
            except Exception as e:
                show_warning(f"ポイントの収集に失敗しました: {e}")
                return

            if len(point_coords) == 0:
                show_info("有効なポイントがありません。")
                return

            try:
                predictor = self._get_sam2_image_predictor()
                image_rgb = self._prepare_rgb_image(image_layer)
                predictor.set_image(image_rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            except Exception as e:
                show_warning(f"SAM2ポイント推論に失敗しました: {e}")
                traceback.print_exc()
                return

            if masks is None or len(masks) == 0:
                show_info("ポイント推論でマスクが得られませんでした。")
                return

            polygons = []
            polygon_scores = []
            for mask, score in zip(masks, scores):
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if contour.shape[0] < 3:
                        continue
                    contour_xy = contour[:, 0, :]
                    polygons.append(contour_xy[:, [1, 0]])
                    polygon_scores.append(float(score))

            if not polygons:
                show_info("ポイント推論結果から有効なポリゴンを作れませんでした。")
                return

            shapes_name = f"{image_layer.name}_sam2_points"
            if any(existing.name == shapes_name for existing in self.viewer.layers):
                del self.viewer.layers[shapes_name]

            shapes_layer = self.viewer.add_shapes(
                polygons,
                shape_type="polygon",
                name=shapes_name,
                properties={"score": np.asarray(polygon_scores)},
                edge_color="#22C55E",
                face_color="transparent",
                edge_width=2,
            )
            self._attach_metadata(
                shapes_layer,
                {
                    "mode": "sam2_point_prompt",
                    "image_layer": image_layer.name,
                    "layer_names": asdict(self._layer_names(image_layer.name)),
                    "output_layer": shapes_name,
                    "polygon_count": len(polygons),
                    "best_score": float(np.max(scores)),
                },
            )
            best_idx = int(np.argmax(scores))
            show_info(
                f"SAM2 point-prompt masks: {len(polygons)} (best mask index={best_idx}, score={float(scores[best_idx]):.3f})"
            )

        self.run_sam2_point_prompt_widget = run_sam2_point_prompt

        @magicgui(
            call_button="Export segmentation artifacts",
            output_dir={"mode": "d"},
        )
        def export_segmentation_artifacts(
            output_dir: Path = Path("./outputs"),
        ) -> None:
            cropped_layer = self._resolve_target_cropped_image_layer()
            if cropped_layer is None:
                show_warning(
                    "対象の cropped 画像レイヤーが見つかりません。先に Crop to ROI を実行してください。"
                )
                return

            sam2_layer = self._resolve_sam2_auto_labels_layer(cropped_layer)
            if sam2_layer is None:
                return

            image_name = (
                cropped_layer.name[: -len("_cropped")]
                if cropped_layer.name.endswith("_cropped")
                else cropped_layer.name
            )
            layer_names = self._layer_names(image_name)

            try:
                export_dir = self._ensure_export_dir(output_dir, image_name)
                sam2_original_layer = self._resolve_sam2_original_labels_layer(
                    cropped_layer
                )
                sam2_labels_source = (
                    sam2_original_layer
                    if sam2_original_layer is not None
                    else sam2_layer
                )
                active_layer = self.viewer.layers.selection.active
                final_layer: Labels
                if (
                    isinstance(active_layer, Labels)
                    and active_layer.data.shape == sam2_layer.data.shape
                    and active_layer.name != sam2_labels_source.name
                ):
                    final_layer = active_layer
                else:
                    final_layer = sam2_layer
                sam2_labels = np.asarray(sam2_labels_source.data, dtype=np.int32)
                final_labels = np.asarray(final_layer.data, dtype=np.int32)
                cropped_rgb = self._prepare_rgb_image(cropped_layer)
                source_layer = self.viewer.layers[layer_names.image]
                if not isinstance(source_layer, Image):
                    raise ValueError(
                        f"{layer_names.image} は画像レイヤーではありません。"
                    )
                source_rgb = self._prepare_rgb_image(source_layer)
                roi_bbox_yx = self._extract_latest_roi_bbox(
                    layer_names.roi, source_rgb.shape[:2]
                )

                sam2_overlay = self._build_overlay_with_ids(cropped_rgb, sam2_labels)
                final_overlay = self._build_overlay_with_ids(cropped_rgb, final_labels)
                roi_overlay = self._build_roi_overlay(source_rgb, roi_bbox_yx)

                cv2.imwrite(
                    str(export_dir / "sam2_only_overlay.png"),
                    cv2.cvtColor(sam2_overlay, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(export_dir / "final_overlay.png"),
                    cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(export_dir / "cropped_image.png"),
                    cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(export_dir / "source_with_roi.png"),
                    cv2.cvtColor(roi_overlay, cv2.COLOR_RGB2BGR),
                )

                self._write_blob_csv(export_dir / "sam2_blobs.csv", sam2_labels)
                self._write_blob_csv(export_dir / "final_blobs.csv", final_labels)

                output_files = sorted(
                    [p.name for p in export_dir.iterdir() if p.is_file()]
                    + ["run_metadata.json"]
                )
                metadata = self._build_export_metadata(
                    image_name=image_name,
                    cropped_layer=cropped_layer,
                    sam2_layer=sam2_labels_source,
                    final_layer=final_layer,
                    sam2_labels=sam2_labels,
                    final_labels=final_labels,
                    roi_bbox_yx=roi_bbox_yx,
                    output_files=output_files,
                )
                (export_dir / "run_metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                show_warning(f"エクスポートに失敗しました: {e}")
                traceback.print_exc()
                return

            show_info(
                f"書き出し完了: {export_dir} (sam2={self._count_positive_labels(sam2_labels)}, final={self._count_positive_labels(final_labels)})"
            )

        self.export_segmentation_artifacts_widget = export_segmentation_artifacts

        @magicgui(call_button="現在画像のレイヤーを全削除")
        def clear_current_image_layers() -> None:
            target_layer_names = [layer.name for layer in list(self.viewer.layers)]
            if not target_layer_names:
                show_info("削除対象のレイヤーが見つかりませんでした。")
                return

            reply = QMessageBox.question(
                self.viewer.window._qt_window,
                "レイヤー削除の確認",
                (
                    f"存在する {len(target_layer_names)} レイヤーをすべて削除します。"
                    "\nこの操作は元に戻せません。続行しますか？"
                ),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Ok:
                show_info("レイヤー削除をキャンセルしました。")
                return

            for layer_name in target_layer_names:
                if layer_name in self.viewer.layers:
                    del self.viewer.layers[layer_name]

            show_info(f"{len(target_layer_names)} レイヤーを削除しました。")

        self.clear_current_image_layers_widget = clear_current_image_layers

    def _ensure_export_dir(self, output_dir: Path, image_name: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(
            ch if ch.isalnum() or ch in "-._" else "_" for ch in image_name
        )
        export_dir = Path(output_dir).expanduser() / f"{safe_name}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def _extract_latest_roi_bbox(
        self, roi_layer_name: str, image_shape: tuple[int, int]
    ) -> list[int] | None:
        if roi_layer_name not in self.viewer.layers:
            return None
        roi_layer = self.viewer.layers[roi_layer_name]
        if len(roi_layer.data) == 0:
            return None

        roi = np.asarray(roi_layer.data[-1])
        ymin = int(np.floor(roi[:, 0].min()))
        ymax = int(np.ceil(roi[:, 0].max()))
        xmin = int(np.floor(roi[:, 1].min()))
        xmax = int(np.ceil(roi[:, 1].max()))

        h, w = image_shape
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        if ymin >= ymax or xmin >= xmax:
            return None
        return [ymin, ymax, xmin, xmax]

    def _label_colormap(self, label: int) -> tuple[int, int, int]:
        seed = int(label * 1103515245 + 12345) & 0x7FFFFFFF
        b = 80 + (seed % 156)
        g = 80 + ((seed // 97) % 156)
        r = 80 + ((seed // 197) % 156)
        return int(r), int(g), int(b)

    def _build_overlay_with_ids(
        self, image_rgb: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        overlay = image_rgb.copy()
        blended = image_rgb.copy()
        unique_labels = [int(v) for v in np.unique(labels) if int(v) > 0]

        for label_id in unique_labels:
            mask = labels == label_id
            if not np.any(mask):
                continue
            color = self._label_colormap(label_id)
            blended[mask] = color

        overlay = cv2.addWeighted(blended, 0.45, image_rgb, 0.55, 0.0)

        for blob in self._compute_blob_metrics(labels):
            cx = int(round(blob["centroid_x"]))
            cy = int(round(blob["centroid_y"]))
            cv2.putText(
                overlay,
                str(blob["id"]),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                str(blob["id"]),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _build_roi_overlay(
        self, source_rgb: np.ndarray, roi_bbox_yx: list[int] | None
    ) -> np.ndarray:
        canvas = source_rgb.copy()
        if roi_bbox_yx is None:
            return canvas
        ymin, ymax, xmin, xmax = roi_bbox_yx
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 140, 0), 3)
        cv2.putText(
            canvas,
            "ROI",
            (xmin, max(ymin - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _compute_blob_metrics(self, labels: np.ndarray) -> list[dict]:
        blobs: list[dict] = []
        unique_labels = [int(v) for v in np.unique(labels) if int(v) > 0]
        for label_id in unique_labels:
            mask = (labels == label_id).astype(np.uint8)
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.where(mask > 0)
            y_min = int(ys.min())
            y_max = int(ys.max())
            x_min = int(xs.min())
            x_max = int(xs.max())
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
            circularity = (
                float((4.0 * np.pi * area) / (perimeter * perimeter))
                if perimeter > 0.0
                else float("nan")
            )
            moments = cv2.moments(mask)
            if moments["m00"] == 0:
                cx = float(xs.mean())
                cy = float(ys.mean())
            else:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])

            blobs.append(
                {
                    "id": label_id,
                    "area_px": area,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "bbox_x_min": x_min,
                    "bbox_y_min": y_min,
                    "bbox_x_max": x_max,
                    "bbox_y_max": y_max,
                    "bbox_width": x_max - x_min + 1,
                    "bbox_height": y_max - y_min + 1,
                    "perimeter_px": perimeter,
                    "circularity": circularity,
                }
            )
        return blobs

    def _write_blob_csv(self, path: Path, labels: np.ndarray) -> None:
        rows = self._compute_blob_metrics(labels)
        fieldnames = [
            "id",
            "area_px",
            "centroid_x",
            "centroid_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
            "bbox_width",
            "bbox_height",
            "perimeter_px",
            "circularity",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _count_positive_labels(self, labels: np.ndarray) -> int:
        unique_labels = np.unique(np.asarray(labels, dtype=np.int32))
        return int(np.count_nonzero(unique_labels > 0))

    def _build_export_metadata(
        self,
        image_name: str,
        cropped_layer: Image,
        sam2_layer: Labels,
        final_layer: Labels,
        sam2_labels: np.ndarray,
        final_labels: np.ndarray,
        roi_bbox_yx: list[int] | None,
        output_files: list[str],
    ) -> dict:
        sam2_payload = dict(getattr(sam2_layer, "metadata", {}) or {}).get(
            "male_flower_count", {}
        )
        sam2_params = sam2_payload.get("sam2_params", {})
        dedup_removed = int(sam2_payload.get("dedup_removed_count", 0))
        sam2_after_dedup = self._count_positive_labels(sam2_labels)
        sam2_raw = sam2_after_dedup + dedup_removed
        final_count = self._count_positive_labels(final_labels)
        manual_net = final_count - sam2_after_dedup

        return {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_image_layer": image_name,
            "source_image_shape_hw": list(
                self.viewer.layers[image_name].data.shape[:2]
            ),
            "cropped_layer_name": cropped_layer.name,
            "cropped_shape_hw": list(cropped_layer.data.shape[:2]),
            "roi_bbox_yx": roi_bbox_yx,
            "sam2_model": {
                "checkpoint_and_cfg": self._get_sam2_checkpoint_and_cfg(),
                "sam2_auto_params": sam2_params,
            },
            "counts": {
                "sam2_raw_count_before_dedup": sam2_raw,
                "sam2_after_dedup_count": sam2_after_dedup,
                "dedup_removed_count": dedup_removed,
                "final_count": final_count,
                "manual_delta_count_net": manual_net,
                "manual_added_estimated": max(0, manual_net),
                "manual_removed_estimated": max(0, -manual_net),
            },
            "layers_used": {
                "sam2_layer": sam2_layer.name,
                "final_layer": final_layer.name,
            },
            "randomness": {
                "python_random_seed_explicitly_set": False,
                "numpy_random_seed_explicitly_set": False,
                "python_random_state_head": list(random.getstate()[1][:5]),
                "numpy_random_state_head": [
                    int(v) for v in np.random.get_state()[1][:5]
                ],
            },
            "runtime_environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "cwd": str(Path.cwd()),
            },
            "output_files": output_files,
        }

    def _attach_metadata(self, layer, payload: dict) -> None:
        existing = dict(getattr(layer, "metadata", {}) or {})
        existing["male_flower_count"] = payload
        layer.metadata = existing

    def _activate_shapes_layer(self, shapes_layer) -> None:
        self.viewer.layers.selection.select_only(shapes_layer)
        shapes_layer.mode = "add_rectangle"

        active = self.viewer.layers.selection.active
        show_info(f"active: {active.name if active is not None else None}")

    def _zoom_to_layer(self, layer: Image) -> None:
        h, w = layer.data.shape[:2]
        show_info(f"{h}, {w}")  # デバッグ用
        self.viewer.camera.center = (h / 2.0, w / 2.0)

    def _prepare_rgb_image(self, layer: Image) -> np.ndarray:
        image = np.asarray(layer.data)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("RGB(A)画像を入力してください。")

        rgb = np.asarray(image[..., :3])
        if rgb.dtype == np.uint8:
            return rgb

        rgb_float = np.asarray(rgb, dtype=np.float32)
        rgb_float = np.nan_to_num(rgb_float, nan=0.0, posinf=255.0, neginf=0.0)

        data_min = float(np.min(rgb_float))
        data_max = float(np.max(rgb_float))

        if data_max <= 1.0 and data_min >= 0.0:
            scaled = rgb_float * 255.0
        elif data_max == data_min:
            scaled = np.zeros_like(rgb_float, dtype=np.float32)
        else:
            scaled = (rgb_float - data_min) * (255.0 / (data_max - data_min))

        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def _get_sam2_checkpoint_and_cfg(self) -> tuple[str, str]:
        project_root = Path(__file__).resolve().parent
        checkpoint = (
            project_root / "sam2_repo" / "checkpoints" / "sam2.1_hiera_large.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                "SAM2 checkpointが見つかりません。scripts/setup_sam2.bat 相当のセットアップを先に実施してください。"
            )
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        return str(checkpoint), model_cfg

    def _get_sam2_mask_generator(
        self,
        points_per_side: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        crop_n_layers: int,
        crop_n_points_downscale_factor: int,
    ):
        key = (
            int(points_per_side),
            float(pred_iou_thresh),
            float(stability_score_thresh),
            int(crop_n_layers),
            int(crop_n_points_downscale_factor),
        )
        if key in self._sam2_mask_generators:
            return self._sam2_mask_generators[key]

        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        checkpoint, model_cfg = self._get_sam2_checkpoint_and_cfg()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        show_info(f"using device for SAM2 auto: {device}")

        sam2_model = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=key[0],
            pred_iou_thresh=key[1],
            stability_score_thresh=key[2],
            crop_n_layers=key[3],
            crop_n_points_downscale_factor=key[4],
            box_nms_thresh=0.55,
            crop_nms_thresh=0.55,
            multimask_output=False,
        )
        self._sam2_mask_generators[key] = mask_generator
        return mask_generator

    def _get_sam2_image_predictor(self):
        if self._sam2_image_predictor is not None:
            return self._sam2_image_predictor

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint, model_cfg = self._get_sam2_checkpoint_and_cfg()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        show_info(f"using device for SAM2 predictor: {device}")

        sam2_model = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        self._sam2_image_predictor = SAM2ImagePredictor(sam2_model)
        return self._sam2_image_predictor

    def _masks_to_label_image(
        self, masks: list[dict], image_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Convert a list of segmentation masks into a label image.

        Parameters
        ----------
        masks : list[dict]
            A list of annotation dictionaries. Each annotation may contain
            a "segmentation" entry representing a boolean mask.
        image_shape : tuple[int, int]
            Shape of the output label image as (height, width).

        Returns
        -------
        np.ndarray
            A 2D integer array where 0 indicates background and positive
            integers indicate mask labels assigned in the order of `masks`
            starting from 1.

        Notes
        -----
        If masks overlap, later masks overwrite earlier ones.
        """
        label_image = np.zeros(image_shape, dtype=np.int32)
        for idx, ann in enumerate(masks, start=1):
            mask = ann.get("segmentation")
            if mask is None:
                continue
            label_image[np.asarray(mask, dtype=bool)] = idx
        return label_image

    def _masks_to_polygons(
        self, masks: list[dict]
    ) -> tuple[list[np.ndarray], list[float]]:
        polygons: list[np.ndarray] = []
        polygon_scores: list[float] = []
        for ann in masks:
            mask = ann.get("segmentation")
            if mask is None:
                continue

            score = float(ann.get("predicted_iou", np.nan))
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                contour_xy = contour[:, 0, :]
                polygons.append(contour_xy[:, [1, 0]])  # y, x for napari
                polygon_scores.append(score)
        return polygons, polygon_scores

    def _sort_masks_by_position(self, masks: list[dict], method: str) -> list[dict]:
        def sort_key(ann: dict) -> tuple[float, float]:
            if method == "bbox":
                bbox = ann.get("bbox")
                if bbox is None or len(bbox) < 2:
                    return (np.inf, np.inf)
                x, y = bbox[0], bbox[1]
                return float(y), float(x)

            if method != "centroid":
                raise ValueError("method must be 'centroid' or 'bbox'")

            mask = ann.get("segmentation")
            if mask is None:
                return (np.inf, np.inf)
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                return (np.inf, np.inf)
            return float(np.mean(ys)), float(np.mean(xs))

        return sorted(masks, key=sort_key)

    def _mask_overlap_stats(
        self, mask_a: np.ndarray, mask_b: np.ndarray
    ) -> dict[str, float]:
        inter = float(np.logical_and(mask_a, mask_b).sum())
        area_a = float(mask_a.sum())
        area_b = float(mask_b.sum())
        union = float(np.logical_or(mask_a, mask_b).sum())
        iou = inter / union if union > 0 else 0.0
        contain = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
        return {
            "intersection": inter,
            "area_a": area_a,
            "area_b": area_b,
            "iou": iou,
            "contain": contain,
        }

    def _deduplicate_masks(
        self,
        masks: list[dict],
        contain_thresh: float = 0.9,
        iou_thresh: float = 0.7,
        score_key: str = "predicted_iou",
    ) -> tuple[list[dict], list[tuple[int, int, dict[str, float]]]]:
        def _score_value(mask: dict) -> float:
            value = mask.get(score_key, 0.0)
            if np.isscalar(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
            return 0.0

        n_masks = len(masks)
        removed: set[int] = set()
        removed_pairs: list[tuple[int, int, dict[str, float]]] = []

        for i in range(n_masks):
            if i in removed:
                continue

            mask_i = masks[i].get("segmentation")
            if mask_i is None:
                continue

            for j in range(i + 1, n_masks):
                if j in removed:
                    continue

                mask_j = masks[j].get("segmentation")
                if mask_j is None:
                    continue

                stats = self._mask_overlap_stats(mask_i, mask_j)
                if stats["contain"] < contain_thresh and stats["iou"] < iou_thresh:
                    continue

                score_i = _score_value(masks[i])
                score_j = _score_value(masks[j])

                if score_i >= score_j:
                    removed.add(j)
                    removed_pairs.append((i, j, stats))
                else:
                    removed.add(i)
                    removed_pairs.append((j, i, stats))
                    break

        kept_masks = [m for idx, m in enumerate(masks) if idx not in removed]
        return kept_masks, removed_pairs

    def _collect_point_prompts(
        self, image_layer: Image
    ) -> tuple[np.ndarray, np.ndarray]:
        fg_points_name = f"{image_layer.name}_prompts_fg"
        bg_points_name = f"{image_layer.name}_prompts_bg"
        legacy_points_name = f"{image_layer.name}_prompts"

        layers_coords_xy: list[np.ndarray] = []
        layers_labels: list[np.ndarray] = []

        def _append_points(
            points_layer: Points, fixed_label: int | None = None
        ) -> None:
            if len(points_layer.data) == 0:
                return

            labels = np.ones((len(points_layer.data),), dtype=np.int32)
            if fixed_label is not None:
                labels.fill(fixed_label)
            elif "label" in points_layer.features:
                labels = np.asarray(points_layer.features["label"], dtype=np.int32)

            valid = np.isin(labels, [0, 1])
            if not np.all(valid):
                invalid_values = np.unique(labels[~valid])
                raise ValueError(
                    f"labelは0か1のみ使用できます。無効値: {invalid_values.tolist()}"
                )

            # napari points data is (y, x), SAM2 expects (x, y)
            coords_yx = np.asarray(points_layer.data)[:, :2]
            coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
            layers_coords_xy.append(coords_xy)
            layers_labels.append(labels)

        has_fg = fg_points_name in self.viewer.layers
        has_bg = bg_points_name in self.viewer.layers
        has_legacy = legacy_points_name in self.viewer.layers

        if has_fg:
            fg_layer = self.viewer.layers[fg_points_name]
            if not isinstance(fg_layer, Points):
                raise ValueError(f"{fg_points_name} はPointsレイヤーではありません。")
            _append_points(fg_layer, fixed_label=1)

        if has_bg:
            bg_layer = self.viewer.layers[bg_points_name]
            if not isinstance(bg_layer, Points):
                raise ValueError(f"{bg_points_name} はPointsレイヤーではありません。")
            _append_points(bg_layer, fixed_label=0)

        if has_legacy:
            points_layer = self.viewer.layers[legacy_points_name]
            if not isinstance(points_layer, Points):
                raise ValueError(
                    f"{legacy_points_name} はPointsレイヤーではありません。"
                )
            _append_points(points_layer)

        if not (has_fg or has_bg or has_legacy):
            raise ValueError(
                f"{fg_points_name} / {bg_points_name} / {legacy_points_name} がありません。Create prompt points layer で作成してください。"
            )

        if not layers_coords_xy:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return (
            np.concatenate(layers_coords_xy, axis=0).astype(np.float32),
            np.concatenate(layers_labels, axis=0).astype(np.int32),
        )

    def _on_active_layer_changed_for_prompt_target(self, _event: Event) -> None:
        self._sync_prompt_target_with_active_layer()

    def _sync_prompt_target_with_active_layer(self) -> None:
        widget = getattr(self, "create_prompt_points_layer_widget", None)
        if widget is None:
            return

        active_layer = self.viewer.layers.selection.active
        if not isinstance(active_layer, Points):
            return

        if active_layer.name.endswith("_prompts_fg"):
            target_value = "Foreground points"
        elif active_layer.name.endswith("_prompts_bg"):
            target_value = "Background points"
        else:
            return

        if widget.prompt_target.value != target_value:
            widget.prompt_target.value = target_value

    def on_image_layer_added(self, event: Event) -> None:
        if self._is_first_image_initialized:
            return

        layer = event.value

        # ROIレイヤーやクロップ後のレイヤーが追加されたときは何もしない
        if layer.name.endswith("_cropped"):
            return

        if not isinstance(layer, Image):
            return

        show_info(f"画像レイヤーが追加されました: {layer.name}")

        shapes_name = self._layer_names(layer.name).roi
        if any(existing.name == shapes_name for existing in self.viewer.layers):
            show_info(f"{shapes_name} はすでに存在します。")
            return

        shapes_layer = self.viewer.add_shapes(
            ndim=layer.ndim,
            name=shapes_name,
            edge_color="#DA702C",
            face_color="transparent",
            edge_width=2,
        )
        self._is_first_image_initialized = True
        self.viewer.layers.events.inserted.disconnect(self.on_image_layer_added)

        # すぐ選ばず、Qtイベントループの次のタイミングで選ぶ
        QTimer.singleShot(0, lambda: self._activate_shapes_layer(shapes_layer))


def main() -> None:
    viewer = napari.Viewer()

    roi_controller = RoiController(viewer)
    # viewer.window.add_dock_widget(roi_controller.analyze_active_widget, area="right")
    viewer.window.add_dock_widget(roi_controller.crop_to_roi_widget, area="right")
    viewer.window.add_dock_widget(
        roi_controller.run_sam2_auto_on_active_widget, area="right"
    )
    viewer.window.add_dock_widget(
        roi_controller.create_prompt_points_layer_widget, area="right"
    )
    viewer.window.add_dock_widget(
        roi_controller.run_sam2_point_prompt_widget, area="right"
    )
    viewer.window.add_dock_widget(
        roi_controller.export_segmentation_artifacts_widget, area="right"
    )
    viewer.window.add_dock_widget(
        roi_controller.clear_current_image_layers_widget, area="right"
    )
    napari.run()


if __name__ == "__main__":
    main()
