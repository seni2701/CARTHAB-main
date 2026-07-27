import napari
from magicgui import magicgui
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box

class NapariVisualisation:
    def __init__(self, center_line_path: str):
        self.center_line_path = center_line_path
        self.dataset_rio: rio.DatasetReader
        self.gdf_buffer: gpd.GeoDataFrame
        self._create_buffer_river_center_line()
        self.viewer = None
        self.results = {"nir": 130, "ndvi": 0.2, "quit": False, "next": False}
        self.ndvi_mask_layer = None
        self.nir_mask_layer = None
        self.rgb_layer = None

    def _create_buffer_river_center_line(self):
        gdf_centerline: gpd.GeoDataFrame = gpd.read_file(self.center_line_path)
        geom: gpd.GeoSeries = gdf_centerline.geometry
        buffer: gpd.GeoSeries = geom.buffer(50)
        gdf_centerline.geometry = buffer
        self.gdf_buffer = gdf_centerline

    def _on_viewer_closed(self):
        self.results["quit"] = True
        self.viewer = None

    def intersect_tif_and_shp(self) -> pd.DataFrame:
        raster_bounds = self.dataset_rio.bounds
        from pyproj import CRS as ProjCRS
        raster_crs = ProjCRS.from_epsg(2947)
        shapefile_crs = self.gdf_buffer.crs
        if shapefile_crs != raster_crs:
            self.gdf_buffer = self.gdf_buffer.to_crs(raster_crs)
        shapefile_bounds = self.gdf_buffer.total_bounds
        shape_geom: gpd.GeoSeries = self.gdf_buffer.geometry
        raster_box = box(*raster_bounds)
        intersection: gpd.GeoSeries = shape_geom.intersection(raster_box)
        intersection = intersection[~intersection.is_empty]
        bounds = intersection.geometry.bounds.values[0]
        
        row_min, col_min = self.dataset_rio.index(np.amin([bounds[0], bounds[2]]),
                                               np.amin([bounds[1], bounds[3]]))
        row_max, col_max = self.dataset_rio.index(np.amax([bounds[0], bounds[2]]),
                                               np.amax([bounds[1], bounds[3]]))
        indexes = [np.amin([row_min, row_max]), np.amin([col_min, col_max]),
                   np.amax([row_min, row_max]), np.amax([col_min, col_max])]
        
        df = pd.DataFrame(data=np.c_[bounds, indexes], columns=["CRS", "Index"])
        df = df.astype({"CRS": float, "Index": int})
        return df

    def get_RGB(self, src: rio.DatasetReader, extent_clip: pd.DataFrame, out_shape: tuple = None) -> np.ndarray:
        window = rio.windows.Window(extent_clip.loc[1, "Index"], extent_clip.loc[0, "Index"],
                                   extent_clip.loc[3, "Index"] - extent_clip.loc[1, "Index"],
                                   extent_clip.loc[2, "Index"] - extent_clip.loc[0, "Index"])
        red = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
        green = src.read(2, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
        blue = src.read(3, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
        return np.dstack((red, green, blue))

    def calibrate_threshold(self, rgb: np.ndarray,
                            ndvi: np.ndarray, nir: np.ndarray,
                            nir_threshold_init: int,
                            ndvi_threshold_init: float):
        
        # Keep track of current data for the widget callback - DO THIS FIRST
        self.current_nir = nir
        self.current_ndvi = ndvi
        self.results["next"] = False
        self.results["quit"] = False
        
        # Reset thresholds in results to initial values for this image
        self.results["nir"] = nir_threshold_init
        self.results["ndvi"] = ndvi_threshold_init

        if self.viewer is None:
            # We don't specify ndisplay=2 as it's default, but we want to make sure the viewer 
            # is initialized with the first image to get a good initial window size.
            self.viewer = napari.Viewer(title="Calibration des seuils (OpenGL/Napari)")
            
            # Add RGB layer
            self.rgb_layer = self.viewer.add_image(rgb, name='RGB', blending='opaque')

            # Add NDVI Mask layer
            initial_ndvi_mask = (ndvi < ndvi_threshold_init).astype(np.float32)
            self.ndvi_mask_layer = self.viewer.add_image(initial_ndvi_mask, name='NDVI Mask',
                                               colormap='green',
                                               opacity=0.4,
                                               blending='additive',
                                               interpolation2d='nearest')

            # Add NIR Mask layer
            initial_nir_mask = (nir < nir_threshold_init).astype(np.float32)
            self.nir_mask_layer = self.viewer.add_image(initial_nir_mask, name='NIR Mask', 
                                             colormap='blue', 
                                             opacity=0.4, 
                                             blending='additive',
                                             interpolation2d='nearest')
            
            # Suggest a larger initial window size if it's too small
            def resize_once():
                if self.viewer and self.viewer.window._qt_window:
                    geom = self.viewer.window._qt_window.geometry()
                    if geom.width() < 1200 or geom.height() < 800:
                        self.viewer.window._qt_window.resize(1200, 900)
            
            from qtpy.QtCore import QTimer
            QTimer.singleShot(100, resize_once)
            
            @magicgui(
                auto_call=True,
                nir_threshold={"widget_type": "Slider", "min": 10, "max": 200, "label": "NIR Threshold"},
                ndvi_threshold={"widget_type": "FloatSlider", "min": -1, "max": 1, "label": "NDVI Threshold"},
                next_image={"widget_type": "PushButton", "text": "Image Suivante (Valider)"},
                quit_app={"widget_type": "PushButton", "text": "Quitter le programme"},
            )
            def update_mask(nir_threshold: int = nir_threshold_init, 
                            ndvi_threshold: float = ndvi_threshold_init,
                            next_image=False,
                            quit_app=False):
                
                # Update masks
                m_nir = (self.current_nir < nir_threshold).astype(np.float32)
                m_ndvi = (self.current_ndvi < ndvi_threshold).astype(np.float32)
                
                self.nir_mask_layer.data = m_nir
                self.ndvi_mask_layer.data = m_ndvi
                
                # Store current values
                self.results["nir"] = nir_threshold
                self.results["ndvi"] = ndvi_threshold

            # No need to set text manually anymore as it's in the decorator

            @update_mask.next_image.clicked.connect
            def on_next_clicked():
                self.results["next"] = True

            @update_mask.quit_app.clicked.connect
            def on_quit_clicked():
                self.results["quit"] = True

            @self.viewer.bind_key('n')
            def next_image_key(viewer):
                self.results["next"] = True

            @self.viewer.bind_key('q')
            def quit_key(viewer):
                self.results["quit"] = True

            # Note: viewer.window.events.closed doesn't exist in all napari versions.
            # We use the Qt signal 'destroyed' on the underlying window to detect closure.
            self.viewer.window._qt_window.destroyed.connect(lambda: self._on_viewer_closed())

            self.viewer.window.add_dock_widget(update_mask, area='bottom')
            self.update_mask_widget = update_mask
        else:
            # Calculer les nouveaux masques
            new_ndvi_mask = (ndvi < ndvi_threshold_init).astype(np.float32)
            new_nir_mask = (nir < nir_threshold_init).astype(np.float32)
            
            # Assigner les nouvelles données
            self.rgb_layer.data = rgb
            self.ndvi_mask_layer.data = new_ndvi_mask
            self.nir_mask_layer.data = new_nir_mask
            
            # Forcer la libération des ressources GPU
            self.viewer.window._qt_viewer.canvas._scene_canvas.context.finish()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset view to fit the new data
            self.viewer.reset_view()

        # Ensure window is visible
        if self.viewer is not None:
            self.viewer.show()

        # Event loop to wait for next image or quit
        import time
        from qtpy.QtWidgets import QApplication
        
        while not self.results["next"] and not self.results["quit"]:
            QApplication.processEvents()
            time.sleep(0.01)
            # Extra safety: check if viewer window is still open
            if self.viewer is None or not hasattr(self.viewer.window, '_qt_window') or not self.viewer.window._qt_window.isVisible():
                self.results["quit"] = True
                break

        return self.results["ndvi"], self.results["nir"], self.results["quit"]
