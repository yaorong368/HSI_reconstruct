import numpy as np
import os
import time
import ica

from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
from functools import partial

import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
import argparse

import sys
sys.path.append('/data/users2/yxiao11/model/satellite_project')
from moduler_gen import *
# from moduler_gen import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def compute_sigma_pixels(wavelength_nm, D_m=3.6, pixel_um=10, focal_length_m=100):
    wavelength_m = wavelength_nm * 1e-9
    theta_rad = 0.25 * wavelength_m / D_m
    theta_arcsec = theta_rad * 206265

    plate_scale = (206265 * pixel_um) / (focal_length_m * 1e6)
    sigma_pixels = theta_arcsec / plate_scale
    return sigma_pixels

# -------------------- SAMPLE GENERATION FUNCTION ------------------

def generate_sample(fake_spectra, image_size=256, camera_pos=[0,0,20]):
    
    
    
    # fake_spectra = generate_fake_spectra(data_type=data_type,
    #                                      material_path = material_path,
    #                                      num_spec=num_spec)
    
    color_list, color_to_material = generate_distinct_color_list(len(fake_spectra))
    
    _, components = make_satellite_with_ids(color_list, color_to_material)
    
    # angles = np.random.randint(0, 360, 3)
    angles = np.array([90., 0., 0.])
    # angles = np.array([0., 0., 0.])
    angles[0] = np.random.uniform(-90,90,1)
    angles[1] = np.random.uniform(-90,90,1)
    angles[2] = np.random.uniform(-90,90,1)
    # angles[0] = np.random.randint(55,125,1)
    # angles[1] = np.random.randint(-60,60,1)
    # angles[2] = np.random.randint(-30,30,1)
    material_mask = rasterize_components_with_depth(
        components, image_size=image_size, camera_pos=camera_pos, angles=angles)
    
    spectral_cube, labels = create_spectral_cube((image_size, image_size), material_mask, fake_spectra)
    
    zoomed_material_mask = rasterize_components_with_depth(
        components, image_size=image_size, camera_pos=[0,0,20], angles=angles)
    
    # zoomed_spectral_cube, _ = create_spectral_cube((128, 128), zoomed_material_mask, fake_spectra)

    return material_mask,spectral_cube, labels, zoomed_material_mask

def simulator(image_size, fake_spectra, camera_pos=[0,0,30]):
    
    material_mask,spectral_cube, labels, zoomed_spectral_cube = generate_sample(
        fake_spectra,
        # material_path = '/data/users2/yxiao11/model/satellite_project/material_spectral',
        image_size=image_size, 
        # data_type=data_type, 
        # num_spec=num_spec, 
        camera_pos=camera_pos
    )
    #######################
#     # Generate random k and b for each sample
#     k = np.random.uniform(0.7, 1.2)  # Example range for k
#     b = np.random.uniform(1, 3)      # Example range for b
    n_slices = spectral_cube.shape[0]
#     # Compute kernel sizes based on the linear formula
#     kernel_sizes = (k * np.arange(5,n_slices+5)/1 + b).astype(int)
#     kernel_sizes[kernel_sizes % 2 == 0] += 1  # Ensure odd kernel sizes
#     #######################
#     # Convert kernel sizes to corresponding sigmas
#     sigmas = kernel_sizes / 2.5  # Adjust this scaling factor as needed   

    sigmas = np.linspace(3.5e-7,2.5e-6,n_slices)
    sigmas = 0.25*sigmas/(3.6*5e-6)*200
    blurred_cube = np.stack(
        [gaussian_filter(spectral_cube[j], 
                                       sigma=sigmas[j], 
                                       mode="mirror") for j in range(n_slices)], 
        axis=0
    ) 

    blurred_cube += np.random.randn(*blurred_cube.shape)*0.02

    return material_mask,spectral_cube, blurred_cube, labels, zoomed_spectral_cube


# ---- Worker Function ----
# Get unique job ID from Slurm


def simulator_worker(i, data_type, fake_spectra, run_forever):
    # Reseed numpy RNG with a unique seed
    seed = (int(time.time() * 1e6) + os.getpid() + i) % (2**32 - 1)
    np.random.seed(seed)

    name = data_type

    material_mask, spectral_cube, blurred_cube, label, zoomed_spectral_cube = simulator(image_size=50, fake_spectra=fake_spectra, camera_pos=[0,0,20])

    if run_forever:
        index = np.random.randint(0, 1000)
        if index == np.random.randint(0, 1000):
            print(f"Process {os.getpid()} - Iteration {i} - Index {index}")
    else:
        index = i
        print(f"Process {os.getpid()} - Iteration {i} - Index {index}")

    # Save the data


    # ###pca trhough spectral dim
    # nb = blurred_cube.reshape(num_spec,-1).T
    # blurred_cube,_,_ = ica.pca_whiten(nb, 50)
    # ###----------------------

    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/mask/{index}.npy', material_mask)
    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/blur_cube/{index}.npy', blurred_cube[0:50])
    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/label/{index}.npy', label)
    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/spectral_cube/{index}.npy', zoomed_spectral_cube)

    if run_forever == False:
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/mask/{index}.npy', material_mask)
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/blur_cube/{index}.npy', blurred_cube[0:50])
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/label/{index}.npy', label)
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/spectral_cube/{index}.npy', zoomed_spectral_cube)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Simulation Runner")
    
    parser.add_argument('--run_forever', type=str2bool, default=True, choices=[True, False],
                        help='Whether to run simulation in infinite loop (True/False)')

    args = parser.parse_args()

    data_type = 'Pristine'
    num_spec = 300
    fake_spectra = generate_fake_spectra(data_type=data_type,
                material_path='/data/users2/yxiao11/model/satellite_project/material_spectral/',
                num_spec=num_spec)
    
    ####---sort spectral for convience
    # Original list of keys (sorted numerically)
    ids = sorted(fake_spectra.keys())

    # Spectra matrix: shape (71, num_bands)
    spectra_matrix = np.stack([fake_spectra[i] for i in ids], axis=0)

    # Correlation matrix + sorting
    corr_matrix = np.abs(np.corrcoef(spectra_matrix))
    corr_strength = corr_matrix.sum(axis=1)
    sorted_indices = np.argsort(-corr_strength)  # Descending

    # Reorder original keys by correlation dominance
    sorted_keys = [ids[i] for i in sorted_indices]
    sorted_fake_spectra = {new_idx+1: fake_spectra[old_key] for new_idx, old_key in enumerate(sorted_keys)}

    fake_spectra = sorted_fake_spectra
    ###---------------


    num_processes = os.cpu_count()
    run_forever = args.run_forever  # Set to True for infinite loop----------------
    total_iterations = 1000  # Used only if run_forever is False

    job_id = os.getpid()
    print(f"Job {job_id} starting with {num_processes} processes")

    worker_func = partial(simulator_worker, data_type=data_type, fake_spectra=fake_spectra, run_forever=run_forever)

    if run_forever:
        def infinite_worker(index_start):
            i = index_start
            with mp.Pool(processes=num_processes) as pool:
                while True:
                    results = pool.imap_unordered(worker_func, range(i, i + num_processes * 10))
                    for _ in tqdm(results, total=num_processes * 10):
                        pass
                    i += num_processes * 10

        infinite_worker(index_start=0)

    else:
        with mp.Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap_unordered(worker_func, range(total_iterations)), total=total_iterations))

        print(f"Job {job_id} complete")

