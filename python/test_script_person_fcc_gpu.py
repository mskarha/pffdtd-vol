##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_script_person_fcc_gpu.py
#
# Description: FCC + GPU run with full volumetric VTKHDF ImageData export
#              (for ParaView FlyingEdges3D / contour / volume rendering).
#
#              The C/CUDA engine writes the entire pressure field as a
#              dense Cartesian VTKHDF time series during the simulation.
#              FCC fold/checkerboard handling happens transparently
#              (unfold + 12-neighbour densify on each snapshot).
#
##############################################################################
from sim_setup import sim_setup

sim_setup(
    model_json_file='../data/models/person/model_export.json',
    draw_backend='mayavi',
    mat_folder='../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
                    'rp_mei_posed_001_30k_mat': 'mv_plasterboard.h5',
                    }, #see build_mats.py to set these material impedances from absorption data
    duration=0.01,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7, #for 1% phase velocity error at fmax
    fmax=4000.0,
    save_folder='../data/sim_data/person/gpu',
    save_folder_gpu='../data/sim_data/person/gpu',
    compress=3, #apply level-3 GZIP compression to larger h5 files
    # --- volumetric VTKHDF ImageData export (read by C/CUDA engine) ---
    vol_export=True,
    vol_snapshot_dt=0.0005,   #0.5 ms between snapshots (~2 kHz frame rate, ~20 frames per 10 ms run)
    vol_gzip_level=3,
    # The receiver-grid path below is no longer required for visualization but
    # is still useful if you want simultaneous WAV/IR export at a few points.
    # Set to False to skip entirely and run a pure-volume export.
    use_receiver_grid=False,
)

#then from '../data/sim_data/person/gpu' folder, run (relative path for default folder structure):
#   ../../../../c_cuda/fdtd_main_gpu_single.x
#
#The engine will write `vol_pressure.vtkhdf` in that same folder.  Override the
#path with the env variable PFFDTD_VOL_PATH=/some/other.vtkhdf if needed.
#
#Open the resulting .vtkhdf in ParaView (5.12+).  ImageData supports
#FlyingEdges3D, Contour, Slice, Volume rendering, and the time slider is
#populated automatically from the embedded Steps/Values dataset.

#then post-process WAV/IR outputs (only needed if use_receiver_grid=True):
#   python -m fdtd.process_outputs --data_dir='../data/sim_data/person/gpu/' --fcut_lowpass 2500.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter='stokes' --save_wav --plot
