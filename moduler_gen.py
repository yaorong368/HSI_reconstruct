import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

import trimesh
# import pyrender
from scipy.spatial.transform import Rotation as R
# from scipy.ndimage import gaussian_filter
# import random


# ------------------- UTILITY FUNCTIONS --------------------------

def generate_distinct_color_list(num_colors: int):
    seen_colors = set()
    color_list = []

    while len(color_list) < num_colors:
        r, g, b = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        color_key = (r, g, b)
        if color_key not in seen_colors:
            seen_colors.add(color_key)
            color_list.append([r, g, b, 255])  # Add alpha channel

    color_to_material = {
        (r, g, b): idx + 1 for idx, (r, g, b, _) in enumerate(color_list)
    }

    return color_list, color_to_material

def resample_to_fixed_length(array, target_len):
    array = np.array(array, dtype=np.float32)
    mask = np.isfinite(array)
    if not np.any(mask):
        return np.zeros(target_len, dtype=np.float32)
    array = array[mask]
    return np.interp(
        np.linspace(0, len(array) - 1, target_len),
        np.arange(len(array)),
        array
    )

def generate_fake_spectra(data_type, material_path, num_spec=50):
    """
    Generate fake spectra from Excel files.

    Args:
        data_type: passed into get_material_id(data_type)
        num_spec: number of points to select per spectrum (not the interval)

    Returns:
        dict: material index to resampled spectrum
    """
#     material_path = '/Users/yaorongxiao/Desktop/satellite_project/material_spectral_copy/'
    file_names = [f for f in os.listdir(material_path) if f.endswith('.xlsx')]

    fake_spectra = {}
    color_index = 1

    for file_name in sorted(file_names):  # sort for consistent ordering
        file_path = os.path.join(material_path, file_name)
        sheet_names = pd.ExcelFile(file_path).sheet_names

        # Move 'glass'-related sheet names to the front
        glass_sheets = [name for name in sheet_names if 'glass' in name.lower()]
        other_sheets = [name for name in sheet_names if 'glass' not in name.lower()]
        ordered_sheets = glass_sheets + other_sheets

        for sheet_name in ordered_sheets:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            material_id = get_material_id(data_type)

            if material_id >= df.shape[1]:
#                 print(f"[WARN] Skipping: '{file_name}' -> '{sheet_name}' (invalid material_id={material_id})")
                continue

            values = df.iloc[:, material_id].tolist()[:-2]

            if not any(pd.notna(values)):
#                 print(f"[EMPTY] Skipping: '{file_name}' -> '{sheet_name}' (all values empty or NaN)")
                continue

            values = list_to_numpy_with_mean(values)
            spectrum = resample_to_fixed_length(values, num_spec)

            fake_spectra[color_index] = spectrum
            color_index += 1

    return dict(sorted(fake_spectra.items(), key=lambda x: x[0]))


# def generate_fake_spectra(data_type, material_path, num_spec=50):
#     """
#     Generate fake spectra from Excel files.

#     Args:
#         data_type: passed into get_material_id(data_type)
#         num_spec: number of points to select per spectrum (not the interval)

#     Returns:
#         dict: material index to resampled spectrum
#     """
# #     material_path = '/Users/yaorongxiao/Desktop/satellite_project/material_spectral_copy/'
#     file_names = [f for f in os.listdir(material_path) if f.endswith('.xlsx')]

#     fake_spectra = {}
#     color_index = 1
#     material_category=[
#         'Aluminum 6061',
#         'Stainless_steel',
#         'Titanium',
#         'CMX',
#         'Kapton HN 1 mil',
#         'Si solar cell']

#     for file_name in sorted(file_names):  # sort for consistent ordering
#         file_path = os.path.join(material_path, file_name)
#         sheet_names = pd.ExcelFile(file_path).sheet_names

#         # Move 'glass'-related sheet names to the front
#         glass_sheets = [name for name in sheet_names if 'glass' in name.lower()]
#         other_sheets = [name for name in sheet_names if 'glass' not in name.lower()]
#         ordered_sheets = glass_sheets + other_sheets
        
#         for sheet_name in ordered_sheets:
            
#             if sheet_name in material_category:
#                 df = pd.read_excel(file_path, sheet_name=sheet_name)
#                 material_id = get_material_id(data_type)

#                 if material_id >= df.shape[1]:
#     #                 print(f"[WARN] Skipping: '{file_name}' -> '{sheet_name}' (invalid material_id={material_id})")
#                     continue

#                 values = df.iloc[:, material_id].tolist()[:-2]

#                 if not any(pd.notna(values)):
#     #                 print(f"[EMPTY] Skipping: '{file_name}' -> '{sheet_name}' (all values empty or NaN)")
#                     continue

#                 values = list_to_numpy_with_mean(values)
#                 spectrum = resample_to_fixed_length(values, num_spec)

#                 fake_spectra[color_index] = spectrum
#                 color_index += 1

#     return dict(sorted(fake_spectra.items(), key=lambda x: x[0]))

def get_intrinsics(fov_y, image_size):
    f = (0.5 * image_size) / np.tan(fov_y / 2)
    cx, cy = image_size / 2, image_size / 2
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def project_vertices(vertices, camera_pose, intrinsics):
    verts_hom = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    verts_cam = (camera_pose @ verts_hom.T).T[:, :3]
    verts_proj = (intrinsics @ verts_cam.T).T
    verts_proj[:, 0] /= verts_proj[:, 2]
    verts_proj[:, 1] /= verts_proj[:, 2]
    return verts_proj[:, :2]

def rotate_mesh_vertices(mesh, rotation_matrix):
    rotated_vertices = (rotation_matrix @ mesh.vertices.T).T
    mesh_rotated = mesh.copy()
    mesh_rotated.vertices = rotated_vertices
    return mesh_rotated

def rasterize_components_with_depth(components, image_size=256, camera_pos=[0,0,25], angles=(0, 0, 0)):
    depth_buffer = np.full((image_size, image_size), np.inf)
    material_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    rotation = R.from_euler('xyz', angles, degrees=True)
    rotation_matrix = rotation.as_matrix()

    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = [0, 0, camera_h]
    camera_pose[:3, 3] = camera_pos
    intrinsics = get_intrinsics(np.radians(60), image_size)

    for mesh, material_id in components:
        rotated_mesh = rotate_mesh_vertices(mesh, rotation_matrix)
        verts_3d = (camera_pose[:3, :3] @ rotated_mesh.vertices.T + camera_pose[:3, 3:4]).T
        verts_2d = project_vertices(rotated_mesh.vertices, camera_pose, intrinsics).astype(np.int32)

        for face in rotated_mesh.faces:
            pts_2d = verts_2d[face]
            pts_3d = verts_3d[face]
            if np.any(pts_2d[:, 0] < 0) or np.any(pts_2d[:, 0] >= image_size):
                continue
            if np.any(pts_2d[:, 1] < 0) or np.any(pts_2d[:, 1] >= image_size):
                continue
            
            mask_poly = np.zeros((image_size, image_size), dtype=np.uint8)
            cv2.fillConvexPoly(mask_poly, pts_2d, 1)
            
            z_avg = np.mean(pts_3d[:, 2])
            
            update_mask = (mask_poly == 1) & (z_avg < depth_buffer)
            material_mask[update_mask] = material_id
            depth_buffer[update_mask] = z_avg
    
    return material_mask

def list_to_numpy_with_mean(data_list):
    """
    Converts a list to a NumPy array, replacing missing values ('--') with the mean of other numbers.
    
    :param data_list: List of numbers (some values may be '--')
    :return: NumPy array with the same shape, missing values replaced by the mean
    """
    # Convert list to NumPy array, treating '--' as NaN
    numeric_array = np.array([float(x) if x != '--' else np.nan for x in data_list], dtype=np.float64)
    
    # Compute mean of valid values (ignoring NaNs)
    mean_value = np.nanmean(numeric_array)
    
    # Replace NaN values with the computed mean
    numeric_array = np.where(np.isnan(numeric_array), mean_value, numeric_array)
    
    return numeric_array

def get_material_id(data_type):
    if data_type == "Pristine":
        material_id = 1
    elif data_type == "Irradiated":
        material_id = 2
    elif data_type == "mixed":
        material_id = np.random.randint(1,3)
    return material_id


def create_spectral_cube(image_shape, material_mask, spectra):
    h, w = image_shape
    freq_count = len(list(spectra.values())[0])
    spectral_cube = np.zeros((freq_count, h, w))

    unique_ids = np.unique(material_mask)
#     print(f"Unique material IDs in mask: {unique_ids}")

    for y in range(h):
        for x in range(w):
            material_id = material_mask[y, x]
            if material_id in spectra:
                spectral_cube[:, y, x] = spectra[material_id]
            else:
                spectral_cube[:, y, x] = 0.0

    return spectral_cube, unique_ids[1:]-1

# -------------------- SATELLITE GENERATION ----------------------

def make_satellite_with_ids(color_list, color_to_material):
    color1_up = color_list[np.random.randint(0, 15)]
    color1_mid = color_list[np.random.randint(15, 30)]
    color1_down = color_list[np.random.randint(30, 45)]
    color2 = color_list[np.random.randint(45, 60)]  # antenna
    color3 = color_list[np.random.randint(60, 65)]  # connector
    color4 = color_list[np.random.randint(65, len(color_list))]  # panel
    # color1_up = color_list[0]
    # color1_mid = color_list[1]
    # color1_down = color_list[2]
    # color2 = color_list[3]  # antenna
    # color3 = color_list[4]  # connector
    # color4 = color_list[5]  # panel

    color1_up = color1_up[:3] + [255]
    color1_mid = color1_mid[:3] + [255]
    color1_down = color1_down[:3] + [255]
    color2 = color2[:3] + [255]
    color4 = color4[:3] + [255]

    components = []

    # Body split into up, mid, down
    h_total = 3.0
    h_third = h_total / 3.0

    # Down part
    body_down = trimesh.creation.cylinder(radius=0.8, height=h_third, sections=20)
    body_down.apply_translation([0, 0, -h_total / 2 + h_third / 2])
    body_down.visual.vertex_colors = np.tile(color1_down, (len(body_down.vertices), 1))
    components.append((body_down, color_to_material[tuple(color1_down[:3])]))

    # Mid part
    body_mid = trimesh.creation.cylinder(radius=0.8, height=h_third, sections=50)
    body_mid.visual.vertex_colors = np.tile(color1_mid, (len(body_mid.vertices), 1))
    components.append((body_mid, color_to_material[tuple(color1_mid[:3])]))

    # Up part
    body_up = trimesh.creation.cylinder(radius=0.8, height=h_third, sections=20)
    body_up.apply_translation([0, 0, h_total / 2 - h_third / 2])
    body_up.visual.vertex_colors = np.tile(color1_up, (len(body_up.vertices), 1))
    components.append((body_up, color_to_material[tuple(color1_up[:3])]))

    # Antenna
    antenna = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    antenna.apply_translation([0, 0, h_total / 2 + 0.8])
    antenna.visual.vertex_colors = np.tile(color2, (len(antenna.vertices), 1))
    components.append((antenna, color_to_material[tuple(color2[:3])]))

    # Connectors
    connector1 = trimesh.creation.box(extents=[1.0, 0.3, 0.3])
    connector1.apply_translation([0.8 + 0.5, 0, 0])
    connector2 = trimesh.creation.box(extents=[1.0, 0.3, 0.3])
    connector2.apply_translation([-0.8 - 0.5, 0, 0])
    connectors = trimesh.util.concatenate([connector1, connector2])
    connectors.visual.vertex_colors = np.tile(color3, (len(connectors.vertices), 1))
    components.append((connectors, color_to_material[tuple(color3[:3])]))

    # Panels
    panel1 = trimesh.creation.box(extents=[3.5, 0.01, 2.0])
    panel1.apply_translation([4.5, 0, 0])
    panel2 = trimesh.creation.box(extents=[3.5, 0.01, 2.0])
    panel2.apply_translation([-4.5, 0, 0])
    solar_panels = trimesh.util.concatenate([panel1, panel2])
    solar_panels.visual.vertex_colors = np.tile(color4, (len(solar_panels.vertices), 1))
    components.append((solar_panels, color_to_material[tuple(color4[:3])]))

    # Combine all
    satellite = trimesh.util.concatenate([
        body_down, body_mid, body_up,
        antenna, connectors, solar_panels
    ])

    return satellite, components


