import napari

from src.services.export_service import ExportService
from src.services.sam2_service import Sam2Service
from src.ui.controller import RoiController


def main() -> None:
    viewer = napari.Viewer()
    roi_controller = RoiController(
        viewer=viewer,
        sam2_service=Sam2Service(),
        export_service=ExportService(),
    )

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
