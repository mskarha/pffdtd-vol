# 3D Wave Propagation Visualization Guide

There are now **two** ways to get a wave-propagation animation into ParaView,
and you almost certainly want path **A**:

| Path | Format | Captures |  Filters available | When to use |
| ---  | ---    | ---     |       ---          |    ---      |
| **A. Volumetric snapshot** (RECOMMENDED) | VTKHDF `ImageData` | The full pressure field at every grid point | FlyingEdges3D, Contour, Slice, Volume Rendering, Threshold | Smooth continuous visualization, isosurfaces |
| B. Receiver-grid (legacy) | VTKHDF `UnstructuredGrid` (vertex cells) | Sparse subset of points where you placed receivers | Glyph, Threshold | When you also want WAV/IR per-point output |

---

## Path A — Volumetric snapshot export (recommended)

The C/CUDA engine writes the **entire pressure field as a dense Cartesian
`vtkImageData` time series** during the simulation, in a single `.vtkhdf`
file.  This is the right input for FlyingEdges3D, Contour, Volume rendering,
and any other filter that needs a dense scalar volume.

The FCC fold/checkerboard packing is undone on every snapshot
(unfold + 12-neighbour densify on the host) so the file you get out is a
straightforward Cartesian volume of shape `(Nx, Nyf, Nz)` at the simulation's
own grid spacing `h`.

### Setup

In your `sim_setup(...)` call add the new keywords:

```python
sim_setup(
    # ... your usual args ...
    fcc_flag=True,
    save_folder='../data/sim_data/person/gpu',
    save_folder_gpu='../data/sim_data/person/gpu',

    # Volumetric VTKHDF ImageData export
    vol_export=True,
    vol_snapshot_dt=0.0005,   # write a snapshot every 0.5 ms wall-clock
                              # (alternatively pass vol_snapshot_stride=N steps)
    vol_gzip_level=3,         # 0 to disable, 3 is a good default
)
```

This adds three small scalars to `sim_consts.h5`:

| Key | Type | Meaning |
| --- | ---  | ---     |
| `vol_export_enabled` | int8  | 1 to enable |
| `vol_snapshot_stride` | int64 | snapshot every N FDTD steps (>=1) |
| `vol_gzip_level`     | int64 | GZIP level 0..9 for the per-step `Pressure` dataset |

These are read by the C engine on startup; the Python engine ignores them.

### Run

Same as before:

```bash
cd ../data/sim_data/person/gpu
../../../../c_cuda/fdtd_main_gpu_single.x
```

The engine writes `vol_pressure.vtkhdf` into the run directory.  Override the
filename or path with environment variables if needed:

```bash
PFFDTD_VOL_PATH=/some/elsewhere/run42.vtkhdf  ../../../../c_cuda/fdtd_main_gpu_single.x
PFFDTD_VOL_STRIDE=20                          ../../../../c_cuda/fdtd_main_gpu_single.x   # override stride
PFFDTD_VOL_EXPORT=1                           ../../../../c_cuda/fdtd_main_gpu_single.x   # force-enable
```

### File-size sanity

`sim_setup` prints an estimate before voxelisation, e.g.:

```
--SIM_SETUP: vol_export ENABLED  stride=12  effective dt=0.500 ms (~2000.0 fps)  gzip=3
--SIM_SETUP: vol_export size estimate: 220.4 MB/snap (raw), ~20 snaps, ~2.15 GB total
```

If the total looks too big, raise `vol_snapshot_dt` (or `vol_snapshot_stride`)
and re-run `sim_setup`.

### Coordinate system

The writer sets `Origin = (xv[0], yv[0], zv[0])`, `Spacing = (h, h, h)` and a
permutation `Direction` matrix so the volume sits in the **true physical
(X, Y, Z) world coordinates of the room**.  No transpose, no axis swap --
the volume overlays directly with the original geometry mesh in ParaView.

### Open in ParaView

ParaView 5.12+ (VTK 9.3+) is required.

1. **File → Open** → select `vol_pressure.vtkhdf`
2. The dataset shows up as `ImageData` with a populated time slider
3. Apply your preferred filter:
   - **Volume rendering**: representation → "Volume", set up a transfer function
   - **Contour / FlyingEdges3D**: gives clean smooth isosurfaces of pressure
   - **Slice**: arbitrary cross-section planes
   - **Clip**: hide everything above a threshold for a half-volume look
4. Use the time slider to scrub or play the animation

### How FCC is handled

When `fcc_flag=True` the GPU engine uses the folded FCC layout described in
`gpu_engine.h`:

- Storage shape `(Nx, Ny_storage, Nz)` where `Ny_storage = Nyf/2 + 1`
- Logical "active" sites: `(ix + iy_logical + iz) % 2 == 0` on a Cartesian
  lattice of shape `(Nx, Nyf, Nz)` at spacing `h`

For each snapshot, `vol_export.h` does:

1. **Gather** every device's interior x-slab D2H into a host buffer of shape
   `(Nx, Ny_storage, Nz)`
2. **Unfold** by parity: storage cell `(ix, iy, iz)` lands at logical
   `(ix, iy, iz)` if `(ix+iy+iz) % 2 == 0`, else at `(ix, Nyf-1-iy, iz)`
3. **Densify** the inactive half-lattice by averaging the 12 nearest FCC
   neighbours -- the same stencil the air kernel uses.  Equivalent to a
   half-step Cartesian interpolation; perfectly fine for visualization.
4. Append to the `Pressure` dataset in the open VTKHDF writer

For Cartesian (`fcc_flag=False`) only the gather is needed; the buffer is
written directly.

### Pulling the file off EC2

```bash
# in the container on EC2
docker compose exec pffdtd bash -lc \
  'cp /src/pffdtd/data/sim_data/person/gpu/vol_pressure.vtkhdf /host-data/'

# locally on WSL
scp -i ~/.ssh/galaxy2-pem.pem \
  ubuntu@<your-ec2-host>:/home/ubuntu/pffdtd-outputs/vol_pressure.vtkhdf \
  /mnt/c/Users/Matt_S/Downloads/vol_pressure.vtkhdf
```

---

## Path B — Receiver-grid export (legacy)

This is the old workflow.  It produces a `vtkUnstructuredGrid` of vertex
cells -- ParaView can render those as point glyphs but **FlyingEdges3D and
contour will not work** because there are no real cells with volume.

This path remains useful if you want WAV/IR outputs at a sparse grid of
receiver positions in addition to (or instead of) volumetric snapshots.

### Overview

The workflow consists of:
1. **Grid Generation**: Generate a 3D grid of receivers matching simulation grid spacing
2. **Boundary Filtering**: Automatically filter receivers that clash with boundary voxels
3. **Simulation**: Run FDTD simulation as normal
4. **Export**: Convert outputs to VTKHDF format for ParaView

## Usage

### Step 1: Setup with Grid-Based Receivers

Modify your setup script to use grid-based receivers:

```python
from sim_setup import sim_setup

sim_setup(
    # ... other parameters ...
    use_receiver_grid=True,  # Enable grid-based receivers
    receiver_grid_spacing=0.1,  # Receiver grid spacing in meters (default: 0.1m)
    receiver_grid_boundary_margin=0.1,  # Minimum distance from boundaries in meters (default: 0.1m)
    # ... rest of parameters ...
)
```

### Step 2: Run Simulation

Run the simulation as normal (Python or CUDA engine).

### Step 3: Post-Process Outputs

Process the outputs with filtering (resampling is optional and only needed for WAV files):

```bash
python -m fdtd.process_outputs --data_dir='../data/sim_data/mv_fcc/gpu/' \
    --fcut_lowpass 2500.0 --N_order_lowpass=8 --symmetric \
    --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter='stokes' \
    --plot
```

**Note:** Resampling is now optional and only happens if:
- You explicitly pass `--resample_Fs <rate>` 
- Or if you use `--save_wav` (auto-resamples to 48kHz for audio)

For VTKHDF export, you don't need resampling - the raw simulation sample rate is fine.

### Step 4: Export to VTKHDF

Export to VTKHDF format for ParaView (follows official VTKHDF specification):

```bash
python -m fdtd.export_vtkhdf --data_dir='../data/sim_data/mv_fcc/gpu/' \
    --output='wave_propagation.vtkhdf'
```

Or from Python:

```python
from fdtd.export_vtkhdf import export_to_vtkhdf

export_to_vtkhdf(
    data_dir='../data/sim_data/mv_fcc/gpu/',
    output_file='wave_propagation.vtkhdf',
    use_processed=True,
    fill_invalid_with_nan=True
)
```

**Note:** The output file will have `.vtkhdf` extension (or `.hdf`), which ParaView recognizes natively.

### Step 5: Generate Coordinate Mapping (Optional)

To understand coordinate transformations:

```bash
python -m fdtd.coordinate_mapping --data_dir='../data/sim_data/mv_fcc/gpu/' \
    --output='coordinate_mapping.csv'
```

This creates a CSV file documenting:
- Real-space coordinates (x, y, z in meters)
- Voxel indices (ix, iy, iz)
- 1D linear indices (ixyz)
- Verification of coordinate transformations

## Coordinate System Reference

### 1D Index Formula
```
ixyz = ix * Ny * Nz + iy * Nz + iz
```

Where:
- `ix, iy, iz` are voxel indices (0-based)
- `Ny, Nz` are grid dimensions
- `z` is the contiguous dimension

### Inverse Transformation
```python
iz = ixyz % Nz
iy = (ixyz - iz) // Nz % Ny
ix = ((ixyz - iz) // Nz - iy) // Ny
```

## File Formats

### Receiver Grid Metadata (`receiver_grid_metadata.json`)

Contains:
- `use_grid`: Boolean indicating grid-based receivers
- `grid_shape`: Tuple (Nx, Ny, Nz) of grid dimensions
- `N_receivers_total`: Total receivers generated
- `N_receivers_valid`: Valid receivers after filtering
- `N_receivers_filtered`: Number filtered out
- `valid_Rxyz`: List of valid receiver positions
- `valid_grid_indices`: List of grid indices for valid receivers

### VTKHDF Output

The exporter creates a single `.vtkhdf` file following the [official VTKHDF specification](https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/) for time-dependent data.

**File Structure:**
- **VTKHDF root group** with Version 2.0 and Type 'UnstructuredGrid'
- **Points**: Grid point coordinates (static, same for all time steps)
- **Cells**: Cell connectivity (static, vertex cells for each point)
- **PointData/Pressure**: Pressure data flattened across all time steps
- **Steps group**: Time-dependent metadata including:
  - `Values`: Time values for each step
  - `PointOffsets`: Offsets for reading points (static geometry)
  - `PointDataOffsets/Pressure`: Offsets for reading pressure data at each time step
  - `CellOffsets` and `ConnectivityIdOffsets`: Offsets for topology (static)

**Key Features:**
- Single file contains all time steps (efficient I/O)
- Uses HDF5 compression (GZIP level 3)
- Follows official VTKHDF 2.0 specification
- ParaView recognizes time series automatically

Invalid receivers (clashing with boundaries) are marked with NaN.

## ParaView Visualization

1. **Open ParaView**
2. **File → Open** → Select the `.vtkhdf` file
3. ParaView should automatically recognize it as a time series
4. **Use the time slider** at the top to animate through all time steps
5. **Apply filters as needed:**
   - **Threshold**: Remove NaN values (set min/max to exclude NaN)
   - **Glyph**: Visualize as arrows/spheres
   - **Slice**: Create cross-sections
   - **Contour**: Isosurfaces
   - **Volume Rendering**: For 3D visualization

**Time Series:**
- The file contains all time steps in a single file
- ParaView's time slider will show all available time steps
- You can play/pause the animation or scrub through time manually

## Notes

- Receiver grid spacing is configurable in meters (default: 0.1m), independent of simulation grid spacing
- Boundary margin defaults to 0.1m (configurable in meters)
- Invalid receivers are marked with NaN (not removed) for visualization
- Coordinate mapping utility helps debug indexing issues
- VTKHDF uses single file with all time steps for efficient I/O

## Troubleshooting

### Receivers Clashing with Boundaries

If you see errors about receivers clashing:
- Increase `receiver_grid_boundary_margin`
- Check that room bounds are correct
- Verify grid spacing is appropriate

### Coordinate Mapping Issues

If you have indexing problems:
- Run `coordinate_mapping.py` to generate reference CSV
- Check that 1D indices match expected formula
- Verify grid dimensions match metadata

### VTKHDF Import Issues

If ParaView can't read the file:
- **Check ParaView version**: VTKHDF support requires ParaView 5.12.0+ or VTK 9.3.0+
- **File extension**: Ensure the file has `.vtkhdf` or `.hdf` extension
- **Verify file structure**: Use `h5dump` or HDFView to inspect the file structure
- **Check Steps group**: Verify that `Steps/Values` contains time values and `Steps/PointDataOffsets/Pressure` contains offsets
- **Update ParaView**: If using an older version, update to the latest release

The exporter follows the official VTKHDF specification, so it should work with compatible ParaView versions.
