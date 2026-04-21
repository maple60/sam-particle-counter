import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import glasbey
from napari.utils import CyclicLabelColormap
from magicgui import magicgui
from napari import Viewer
from napari.layers import Image, Labels, Points
from napari.utils.events import Event
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from src.services.export_service import ExportService
from src.services.sam2_service import Sam2Service


class RoiController:
    def __init__(
        self,
        viewer: Viewer,
        sam2_service: Sam2Service,
        export_service: ExportService,
    ):
        self.viewer = viewer
        self.sam2_service = sam2_service
        self.export_service = export_service
        self._sam2_id_zoom_handlers: dict[str, Callable] = {}
        self._sam2_id_points_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._is_first_image_initialized = False
        self._ensure_image_layer_insert_listener()
        self._ensure_layer_remove_listener()
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
        sam2_auto_ids: str
        sam2_auto_ids_outline: str
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
            sam2_auto_ids=f"{image_layer_name}_sam2_auto_ids",
            sam2_auto_ids_outline=f"{image_layer_name}_sam2_auto_ids_outline",
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
            "_sam2_auto_ids",
            "_sam2_auto_ids_outline",
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
                show_warning(f"{layer_name} is not a Labels layer.")
                return None

            if layer.data.shape[:2] != cropped_layer.data.shape[:2]:
                continue

            return layer

        show_warning(
            (
                f"Could not find {candidate_names[0]} or {candidate_names[1]}."
                "Run SAM2 auto segmentation on the cropped image first."
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
        @magicgui(call_button="Crop to ROI")
        def crop_to_roi() -> None:
            layer = self.viewer.layers.selection.active
            if not layer.name.endswith("_ROI"):
                show_info("Please select an ROI layer.")
                return

            image_layer_name = layer.name[:-4]
            layer_names = self._layer_names(image_layer_name)
            try:
                image_layer = self.viewer.layers[layer_names.image]
            except KeyError:
                show_warning(f"Could not find an image layer named {image_layer_name}.")
                return
            if not isinstance(image_layer, Image):
                show_warning(f"{image_layer_name} is not an image layer.")
                return
            show_info("Resolved the corresponding image layer.")

            if len(layer.data) == 0:
                show_warning("No ROI found. Please draw one rectangle.")
                return
            roi = layer.data[-1]
            show_info(f"ROI vertices: {roi}")
            ymin = int(np.floor(roi[:, 0].min()))
            ymax = int(np.ceil(roi[:, 0].max()))
            xmin = int(np.floor(roi[:, 1].min()))
            xmax = int(np.ceil(roi[:, 1].max()))
            h, w = image_layer.data.shape[:2]
            ymin = max(0, ymin)
            ymax = min(h, ymax)
            xmin = max(0, xmin)
            xmax = min(w, xmax)

            if ymin >= ymax or xmin >= xmax:
                show_warning("No valid ROI is drawn.")
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
            show_info(f"cropped: y={ymin}:{ymax}, x={xmin}:{xmax}")

            self.viewer.layers.selection.select_only(cropped_layer)
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
                show_warning("Please select exactly one image layer.")
                return

            try:
                mask_generator = self.sam2_service._get_sam2_mask_generator(
                    points_per_side=points_per_side,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    crop_n_layers=crop_n_layers,
                    crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                )
                image_rgb = self.sam2_service.prepare_rgb_image(layer)
                masks = mask_generator.generate(image_rgb)
            except Exception as e:
                show_warning(f"SAM2 auto segmentation failed: {e}")
                traceback.print_exc()
                return

            masks = self.sam2_service._sort_masks_by_position(masks, method="centroid")
            removed_pairs: list[tuple[int, int, dict[str, float]]] = []
            if deduplicate_masks:
                masks, removed_pairs = self.sam2_service._deduplicate_masks(
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
                polygons, polygon_scores = self.sam2_service._masks_to_polygons(masks)
                if not polygons:
                    show_info("No valid mask was produced by SAM2.")
                    return

            label_image: np.ndarray | None = None
            if needs_labels:
                label_image = self.sam2_service._masks_to_label_image(
                    masks, layer.data.shape[:2]
                )
                if np.max(label_image) == 0:
                    show_info("No valid mask was produced by SAM2.")
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
                    edge_color="#5E409D",
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

                palette_glasbey = glasbey.create_palette(
                    palette_size=12, colorblind_safe=True
                )
                cmap_glasbey = CyclicLabelColormap(
                    colors=["transparent"] + palette_glasbey
                )
                labels_layer = self.viewer.add_labels(
                    label_image, name=labels_name, opacity=0.8, colormap=cmap_glasbey
                )
                original_labels_layer = self.viewer.add_labels(
                    label_image.copy(),
                    name=original_labels_name,
                    opacity=0.0,
                    visible=False,
                    colormap=cmap_glasbey,
                )
                original_labels_layer.editable = False
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
                self._create_or_update_sam2_id_layers(
                    labels_layer, min_area=0
                )
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
                    "Could not find the target cropped image layer. Run Crop to ROI first."
                )
                return

            fg_points_name = f"{layer.name}_prompts_fg"
            bg_points_name = f"{layer.name}_prompts_bg"
            layer_names = self._layer_names(layer.name)

            if bg_points_name in self.viewer.layers:
                bg_layer = self.viewer.layers[bg_points_name]
                if not isinstance(bg_layer, Points):
                    show_warning(f"{bg_points_name} is not a Points layer.")
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
                    show_warning(f"{fg_points_name} is not a Points layer.")
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
                "Prompt point layers are ready. Add points to the selected target from the dropdown (foreground=green, background=red), then run SAM2 point prompt."
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
                    "Could not find the target cropped image layer. Run Crop to ROI first."
                )
                return

            try:
                point_coords, point_labels = self.sam2_service._collect_point_prompts(
                    image_layer, self.viewer.layers
                )
            except Exception as e:
                show_warning(f"Failed to collect points: {e}")
                return

            if len(point_coords) == 0:
                show_info("No valid points found.")
                return

            try:
                predictor = self.sam2_service._get_sam2_image_predictor()
                image_rgb = self.sam2_service.prepare_rgb_image(image_layer)
                predictor.set_image(image_rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            except Exception as e:
                show_warning(f"SAM2 point inference failed: {e}")
                traceback.print_exc()
                return

            if masks is None or len(masks) == 0:
                show_info("No mask was produced from point inference.")
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
                show_info(
                    "Could not create valid polygons from point inference results."
                )
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
                    "Could not find the target cropped image layer. Run Crop to ROI first."
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
                export_dir = self.export_service.ensure_export_dir(
                    output_dir, image_name
                )
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
                cropped_rgb = self.sam2_service.prepare_rgb_image(cropped_layer)
                source_layer = self.viewer.layers[layer_names.image]
                if not isinstance(source_layer, Image):
                    raise ValueError(f"{layer_names.image} is not an image layer.")
                source_rgb = self.sam2_service.prepare_rgb_image(source_layer)
                roi_bbox_yx = None
                if layer_names.roi in self.viewer.layers:
                    roi_bbox_yx = self.export_service.extract_latest_roi_bbox(
                        self.viewer.layers[layer_names.roi], source_rgb.shape[:2]
                    )

                sam2_overlay = self.export_service._build_overlay_with_ids(
                    cropped_rgb, sam2_labels
                )
                final_overlay = self.export_service._build_overlay_with_ids(
                    cropped_rgb, final_labels
                )
                roi_overlay = self.export_service._build_roi_overlay(
                    source_rgb, roi_bbox_yx
                )

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

                self.export_service._write_blob_csv(
                    export_dir / "sam2_blobs.csv", sam2_labels
                )
                self.export_service._write_blob_csv(
                    export_dir / "final_blobs.csv", final_labels
                )

                output_files = sorted(
                    [p.name for p in export_dir.iterdir() if p.is_file()]
                    + ["run_metadata.json"]
                )
                metadata = self.export_service._build_export_metadata(
                    image_name=image_name,
                    source_layer=source_layer,
                    cropped_layer=cropped_layer,
                    sam2_layer=sam2_labels_source,
                    final_layer=final_layer,
                    sam2_labels=sam2_labels,
                    final_labels=final_labels,
                    roi_bbox_yx=roi_bbox_yx,
                    output_files=output_files,
                    sam2_checkpoint_cfg_getter=self.sam2_service._get_sam2_checkpoint_and_cfg,
                )
                (export_dir / "run_metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                show_warning(f"Export failed: {e}")
                traceback.print_exc()
                return

            show_info(
                f"Export completed: {export_dir} (sam2={self.export_service._count_positive_labels(sam2_labels)}, final={self.export_service._count_positive_labels(final_labels)})"
            )

        self.export_segmentation_artifacts_widget = export_segmentation_artifacts

        @magicgui(call_button="Delete all layers for current image")
        def clear_current_image_layers() -> None:
            target_layer_names = [layer.name for layer in list(self.viewer.layers)]
            if not target_layer_names:
                show_info("No layers found to delete.")
                return

            reply = QMessageBox.question(
                self.viewer.window._qt_window,
                "Confirm layer deletion",
                (
                    f"This will delete all {len(target_layer_names)} existing layers."
                    "\nThis action cannot be undone. Continue?"
                ),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Ok:
                show_info("Layer deletion was canceled.")
                return

            for layer_name in target_layer_names:
                if layer_name in self.viewer.layers:
                    del self.viewer.layers[layer_name]

            self._is_first_image_initialized = False
            self._ensure_image_layer_insert_listener()
            show_info(f"Deleted {len(target_layer_names)} layers.")

        self.clear_current_image_layers_widget = clear_current_image_layers

    def _attach_metadata(self, layer, payload: dict) -> None:
        existing = dict(getattr(layer, "metadata", {}) or {})
        existing["male_flower_count"] = payload
        layer.metadata = existing

    def _activate_shapes_layer(self, shapes_layer) -> None:
        self.viewer.layers.selection.select_only(shapes_layer)
        shapes_layer.mode = "add_rectangle"

        active = self.viewer.layers.selection.active
        show_info(f"active: {active.name if active is not None else None}")

    def _id_text_size_from_zoom(self) -> float:
        zoom = float(self.viewer.camera.zoom)
        return float(np.clip(8.0 + 2.0 * np.log2(max(zoom, 1e-3)), 7.0, 20.0))

    def _compute_label_centroids(
        self, label_image: np.ndarray, min_area: int
    ) -> tuple[np.ndarray, np.ndarray]:
        centroids: list[tuple[float, float]] = []
        label_ids: list[int] = []
        max_id = int(np.max(label_image))
        for label_id in range(1, max_id + 1):
            ys, xs = np.where(label_image == label_id)
            if len(xs) == 0:
                continue
            if min_area > 0 and len(xs) < min_area:
                continue
            centroids.append((float(np.mean(ys)), float(np.mean(xs))))
            label_ids.append(label_id)
        return np.asarray(centroids, dtype=np.float32), np.asarray(label_ids, dtype=np.int32)

    def _create_or_update_sam2_id_layers(self, labels_layer: Labels, min_area: int) -> None:
        labels_name = labels_layer.name
        image_name = labels_name[: -len("_sam2_auto_labels")]
        layer_names = self._layer_names(image_name)
        coords, label_ids = self._compute_label_centroids(
            np.asarray(labels_layer.data, dtype=np.int32), min_area=min_area
        )
        if len(coords) == 0:
            show_warning("No labels found for ID overlay.")
            return
        self._sam2_id_points_cache[image_name] = (coords, label_ids)

        for name in (layer_names.sam2_auto_ids, layer_names.sam2_auto_ids_outline):
            if name in self.viewer.layers:
                del self.viewer.layers[name]

        text_size = self._id_text_size_from_zoom()
        text_strings = np.asarray([str(v) for v in label_ids], dtype=object)
        transparent = np.zeros((len(coords), 4), dtype=np.float32)
        outline_layer = self.viewer.add_points(
            data=coords,
            name=layer_names.sam2_auto_ids_outline,
            features={"id": text_strings},
            size=0.1,
            face_color=transparent,
            border_color=transparent,
            text={"string": "{id}", "color": "black", "size": text_size + 2},
        )
        id_layer = self.viewer.add_points(
            data=coords,
            name=layer_names.sam2_auto_ids,
            features={"id": text_strings},
            size=0.1,
            face_color=transparent,
            border_color=transparent,
            text={"string": "{id}", "color": "white", "size": text_size},
        )

        self._attach_metadata(id_layer, {"mode": "sam2_id_overlay", "display_mode": "always"})
        self._attach_metadata(
            outline_layer, {"mode": "sam2_id_overlay", "display_mode": "always"}
        )
        self._sync_sam2_id_layers_to_zoom(image_name)
        self._set_sam2_id_display_mode(image_name)
        self.viewer.layers.selection.select_only(labels_layer)
        show_info(f"SAM2 ID overlay created: mode=always, labels={len(label_ids)}")

    def _set_sam2_id_display_mode(self, image_name: str) -> None:
        layer_names = self._layer_names(image_name)
        if layer_names.sam2_auto_ids not in self.viewer.layers:
            return
        id_layer = self.viewer.layers[layer_names.sam2_auto_ids]
        outline_layer = self.viewer.layers[layer_names.sam2_auto_ids_outline]
        if not isinstance(id_layer, Points) or not isinstance(outline_layer, Points):
            return

        id_layer.visible = True
        outline_layer.visible = True
        coords, label_ids = self._sam2_id_points_cache[image_name]
        text_strings = np.asarray([str(v) for v in label_ids], dtype=object)
        id_layer.data = coords
        outline_layer.data = coords
        id_layer.features = {"id": text_strings}
        outline_layer.features = {"id": text_strings}

    def _sync_sam2_id_layers_to_zoom(self, image_name: str) -> None:
        layer_names = self._layer_names(image_name)

        def _on_zoom(_event=None) -> None:
            if layer_names.sam2_auto_ids not in self.viewer.layers:
                return
            id_layer = self.viewer.layers[layer_names.sam2_auto_ids]
            outline_layer = self.viewer.layers[layer_names.sam2_auto_ids_outline]
            if not isinstance(id_layer, Points) or not isinstance(outline_layer, Points):
                return
            base_size = self._id_text_size_from_zoom()
            id_layer.text.size = base_size
            outline_layer.text.size = base_size + 2

        if image_name in self._sam2_id_zoom_handlers:
            old = self._sam2_id_zoom_handlers[image_name]
            self.viewer.camera.events.zoom.disconnect(old)
        self._sam2_id_zoom_handlers[image_name] = _on_zoom
        self.viewer.camera.events.zoom.connect(_on_zoom)
        _on_zoom()

    def _cleanup_sam2_id_state(self, image_name: str) -> None:
        zoom_handler = self._sam2_id_zoom_handlers.pop(image_name, None)
        if zoom_handler is not None:
            self.viewer.camera.events.zoom.disconnect(zoom_handler)

        self._sam2_id_points_cache.pop(image_name, None)

    def _zoom_to_layer(self, layer: Image) -> None:
        h, w = layer.data.shape[:2]
        show_info(f"{h}, {w}")
        self.viewer.camera.center = (h / 2.0, w / 2.0)

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

    def _ensure_image_layer_insert_listener(self) -> None:
        events = self.viewer.layers.events.inserted
        events.connect(self.on_image_layer_added)

    def _disable_image_layer_insert_listener(self) -> None:
        events = self.viewer.layers.events.inserted
        events.disconnect(self.on_image_layer_added)

    def _ensure_layer_remove_listener(self) -> None:
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

    def _on_layer_removed(self, event: Event) -> None:
        layer = event.value
        layer_name = str(getattr(layer, "name", ""))
        image_name = self._infer_image_name_from_layer(layer)
        if image_name is None:
            return

        layer_names = self._layer_names(image_name)
        if layer_name not in {layer_names.sam2_auto_ids, layer_names.sam2_auto_ids_outline}:
            return

        self._cleanup_sam2_id_state(image_name)

    def on_image_layer_added(self, event: Event) -> None:
        if self._is_first_image_initialized:
            return

        layer = event.value

        if layer.name.endswith("_cropped"):
            return

        if not isinstance(layer, Image):
            return

        show_info(f"Image layer added: {layer.name}")

        shapes_name = self._layer_names(layer.name).roi
        if any(existing.name == shapes_name for existing in self.viewer.layers):
            show_info(f"{shapes_name} already exists.")
            return

        shapes_layer = self.viewer.add_shapes(
            ndim=layer.ndim,
            name=shapes_name,
            edge_color="#DA702C",
            face_color="transparent",
            edge_width=2,
        )
        self._is_first_image_initialized = True
        self._disable_image_layer_insert_listener()

        QTimer.singleShot(0, lambda: self._activate_shapes_layer(shapes_layer))
