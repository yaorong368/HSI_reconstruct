import numpy as np
import os
import time
# import ica

# from scipy.spatial.transform import Rotation as R
# from scipy.ndimage import gaussian_filter
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


# ---- Worker Function ----
# Get unique job ID from Slurm


def simulator_worker(i, num_spectrum, data_type, fake_material, run_forever):
    # Reseed numpy RNG with a unique seed
    seed = (int(time.time() * 1e6) + os.getpid() + i) % (2**32 - 1)
    np.random.seed(seed)

    name = data_type

    

    if run_forever:
        index = np.random.randint(0, 1000)
        if index == np.random.randint(0, 1000):
            print(f"Process {os.getpid()} - Iteration {i} - Index {index}")
    else:
        index = i
        print(f"Process {os.getpid()} - Iteration {i} - Index {index}")

    # std = np.random.uniform(0.05,0.08)
    # image_size = np.random.choice([16,32])
    # height = np.random.choice([10,15,16,17,20,21,22,30])
    image_size=15
    height=17 # camera position
    zoom_level=1
    std = 0.01
    
    if index < 10000:
        material_mask, spectral_cube, blurred_cube, label, zoomed_spectral_cube = simulator(num_spectrum, 
                                                                                            image_size=image_size, 
                                                                                            fake_material=fake_material, 
                                                                                            camera_pos=[0,0,height],
                                                                                            noise_std=std,
                                                                                            zoom_level=zoom_level)
    elif index < 600:
        material_mask, zoomed_spectral_cube, blurred_cube = shape_simulator(fake_material,
                                                                            size=image_size,
                                                                            num_channels=num_spectrum, 
                                                                            noise_std=std,
                                                                            zoom=zoom_level)
    else:
        material_mask, zoomed_spectral_cube, blurred_cube = shape_simulator2(fake_material,
                                                                            size=image_size,
                                                                            num_channels=num_spectrum, 
                                                                            noise_std=std,
                                                                            num_patches=(12,15),
                                                                            zoom=zoom_level)

    # Save the data

    # ###----------------------

    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/mask/{index}.npy', material_mask)
    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/blur_cube/{index}.npy', blurred_cube)
    # np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/label/{index}.npy', label)
    np.save(f'/data/users2/yxiao11/model/satellite_project/database/{name}/spectral_cube/{index}.npy', zoomed_spectral_cube)

    if run_forever == False:
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/mask/{index}.npy', material_mask)
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/blur_cube/{index}.npy', blurred_cube)
        # np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/label/{index}.npy', label)
        np.save(f'/data/users2/yxiao11/model/satellite_project/data/{name}/spectral_cube/{index}.npy', zoomed_spectral_cube)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Simulation Runner")
    
    parser.add_argument('--run_forever', type=str2bool, default=True, choices=[True, False],
                        help='Whether to run simulation in infinite loop (True/False)')

    args = parser.parse_args()

    data_type = 'Pristine'
    num_spec = 36
    fake_material = generate_material(n=num_spec,
                # material_path='/data/users2/yxiao11/model/satellite_project/material_spectral/',
                end=1000)
    

    ###---------------


    num_processes = os.cpu_count()
    run_forever = args.run_forever  # Set to True for infinite loop----------------
    total_iterations = 1000  # Used only if run_forever is False

    job_id = os.getpid()
    print(f"Job {job_id} starting with {num_processes} processes")

    worker_func = partial(simulator_worker, num_spectrum=num_spec, data_type=data_type, fake_material=fake_material, run_forever=run_forever)

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

