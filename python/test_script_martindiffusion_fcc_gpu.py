##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_script_martindiffusion_fcc_gpu.py
#
# Description: FCC + GPU run with full volumetric VTKHDF ImageData export
#              (for ParaView FlyingEdges3D / contour / volume rendering).
#
##############################################################################
from sim_setup import sim_setup

sim_setup(
    model_json_file='../data/models/martindiffusion/model_export.json',
    draw_backend='mayavi',
    mat_folder='../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
                    'C02_Golden_Beige': 'mv_plasterboard.h5',
                    }, #see build_mats.py to set these material impedances from absorption data
    duration=0.01,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7, #for 1% phase velocity error at fmax
    fmax=4000.0,
    save_folder='../data/sim_data/martindiffusion/gpu',
    save_folder_gpu='../data/sim_data/martindiffusion/gpu',
    compress=3, #apply level-3 GZIP compression to larger h5 files
    # --- volumetric VTKHDF ImageData export (read by C/CUDA engine) ---
    vol_export=True,
    vol_snapshot_dt=0.0005,   #0.5 ms between snapshots
    vol_gzip_level=3,
    use_receiver_grid=False,
)

#then from '../data/sim_data/martindiffusion/gpu' folder, run:
#   ../../../../c_cuda/fdtd_main_gpu_single.x
#
#The engine writes `vol_pressure.vtkhdf` into that folder.  Open in ParaView 5.12+.
#
#WAV/IR post-processing (only needed if use_receiver_grid=True):
#   python -m fdtd.process_outputs --data_dir='../data/sim_data/martindiffusion/gpu/' --fcut_lowpass 2500.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter='stokes' --save_wav --plot
