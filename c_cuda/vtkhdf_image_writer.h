// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// VTKHDF ImageData writer (append-only time series)
//
// Writes a single .vtkhdf file with the VTKHDF 2.3 layout for ImageData,
// suitable for ParaView 5.12+ (VTK 9.3+).  Each call to
// vtkhdf_image_writer_append_step_f32() appends one snapshot to the time
// series, sharing a single static geometry.
//
// Coordinate / ordering convention used by this writer
// ----------------------------------------------------
// The VTKHDF spec stores point data as a flat 1D HDF5 array of length
// (nx * ny * nz).  When viewed as a 3D array the layout is (k, j, i) with i
// (the "X" axis) varying fastest, then j (the "Y" axis), then k (the "Z" axis)
// varying slowest.  World position of grid index (i, j, k) with identity
// Direction is:
//
//     world = Origin + (i*spacing_x, j*spacing_y, k*spacing_z)
//
// In PFFDTD's GPU/CPU engines the contiguous (innermost, fastest) physical
// dimension is **physical Z**, while the slowest is **physical X**.  The flat
// array therefore naturally matches the writer's expected layout if we map:
//
//     writer "X" axis  <-->  physical Z   (fastest, contiguous)
//     writer "Y" axis  <-->  physical Y   (middle)
//     writer "Z" axis  <-->  physical X   (slowest)
//
// To make the resulting volume sit at the true physical (X, Y, Z) world
// coordinates of the room (so it overlays correctly with the original mesh in
// ParaView), the caller should pass:
//
//     nx           = Nz_phys    ny           = Ny_phys    nz           = Nx_phys
//     origin_x     = xv[0]      origin_y     = yv[0]      origin_z     = zv[0]
//     spacing_*    = h
//     direction    = { 0,0,1,  0,1,0,  1,0,0 }   // permute writer axes -> world XYZ
//
// With that Direction the world position of writer-grid index (i, j, k) becomes
// (xv[0] + k*h, yv[0] + j*h, zv[0] + i*h), which is exactly
// (xv[i_phys_x], yv[i_phys_y], zv[i_phys_z]) when i_phys_x = k, i_phys_y = j,
// i_phys_z = i -- i.e., the GPU's natural flat order.  No transpose needed.
//
// Pass `direction = NULL` to use identity (the data will then be displayed in
// (Z,Y,X) world axes).
///////////////////////////////////////////////////////////////////////////////

#ifndef _VTKHDF_IMAGE_WRITER_H
#define _VTKHDF_IMAGE_WRITER_H

#ifndef _STDINT_H
#include <stdint.h>
#endif
#ifndef _STDBOOL_H
#include <stdbool.h>
#endif
#ifndef _STDIO_H
#include <stdio.h>
#endif
#ifndef _STDLIB_H
#include <stdlib.h>
#endif
#ifndef _STRING_H
#include <string.h>
#endif

#include "hdf5.h"

typedef struct VtkhdfImageWriter {
   hid_t file;
   hid_t root;
   hid_t steps;
   hid_t pointData;

   hid_t ds_values;
   hid_t ds_pressure;
   hid_t ds_pressure_offsets;

   int64_t nsteps;
   int64_t npoints;

   // For convenience / debugging
   int64_t nx;
   int64_t ny;
   int64_t nz;
} VtkhdfImageWriter;

static inline void vtkhdf__herr_ok(herr_t status, const char *what) {
   if (status < 0) {
      fprintf(stderr, "VTKHDF writer error: %s\n", what);
      abort();
   }
}

static inline hid_t vtkhdf__create_1d_resizable(hid_t parent, const char *name, hid_t dtype,
                                                hsize_t chunk0, int gzip_level) {
   hsize_t dims[1] = {0};
   hsize_t maxdims[1] = {H5S_UNLIMITED};
   hid_t space = H5Screate_simple(1, dims, maxdims);
   if (space < 0) abort();

   hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
   if (dcpl < 0) abort();
   hsize_t chunk[1] = {chunk0 > 0 ? chunk0 : 1024};
   vtkhdf__herr_ok(H5Pset_chunk(dcpl, 1, chunk), "H5Pset_chunk");
   if (gzip_level > 0) {
      // Shuffle then GZIP -- big win for spatially-coherent float32 fields
      vtkhdf__herr_ok(H5Pset_shuffle(dcpl), "H5Pset_shuffle");
      vtkhdf__herr_ok(H5Pset_deflate(dcpl, (unsigned)gzip_level), "H5Pset_deflate");
   }

   hid_t dset = H5Dcreate(parent, name, dtype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
   if (dset < 0) abort();

   H5Pclose(dcpl);
   H5Sclose(space);
   return dset;
}

static inline void vtkhdf__append_1d(hid_t dset, hid_t dtype, const void *data, hsize_t n) {
   if (n == 0) return;

   hid_t fspace = H5Dget_space(dset);
   if (fspace < 0) abort();

   hsize_t cur_dims[1];
   H5Sget_simple_extent_dims(fspace, cur_dims, NULL);
   H5Sclose(fspace);

   hsize_t new_dims[1] = {cur_dims[0] + n};
   vtkhdf__herr_ok(H5Dset_extent(dset, new_dims), "H5Dset_extent");

   hid_t filespace = H5Dget_space(dset);
   if (filespace < 0) abort();
   hsize_t start[1] = {cur_dims[0]};
   hsize_t count[1] = {n};
   vtkhdf__herr_ok(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL),
                   "H5Sselect_hyperslab");

   hid_t memspace = H5Screate_simple(1, count, NULL);
   if (memspace < 0) abort();

   vtkhdf__herr_ok(H5Dwrite(dset, dtype, memspace, filespace, H5P_DEFAULT, data), "H5Dwrite");

   H5Sclose(memspace);
   H5Sclose(filespace);
}

static inline void vtkhdf__write_string_attr(hid_t parent, const char *name, const char *value) {
   hid_t t = H5Tcopy(H5T_C_S1);
   H5Tset_size(t, strlen(value));
   hid_t s = H5Screate(H5S_SCALAR);
   hid_t a = H5Acreate(parent, name, t, s, H5P_DEFAULT, H5P_DEFAULT);
   vtkhdf__herr_ok(H5Awrite(a, t, value), "H5Awrite string attr");
   H5Aclose(a);
   H5Sclose(s);
   H5Tclose(t);
}

// Open writer.
//   path         : output filename (use .vtkhdf or .hdf for ParaView auto-detect)
//   nx, ny, nz   : dimensions of the writer grid (nx fastest, nz slowest in flat layout)
//   origin_*     : world origin coordinates of voxel (0,0,0)
//   spacing_*    : world spacing of writer grid axes
//   direction    : 9-element row-major 3x3 matrix mapping (writer X, Y, Z) increments
//                  to world (X, Y, Z) increments.  Pass NULL for identity.
//   gzip_level   : 0 to disable, 1..9 for GZIP compression of the per-step Pressure
//                  array (uses HDF5 SHUFFLE + DEFLATE filters).  3 is a good default.
static inline void vtkhdf_image_writer_open(
   VtkhdfImageWriter *w,
   const char *path,
   int64_t nx,
   int64_t ny,
   int64_t nz,
   double origin_x,
   double origin_y,
   double origin_z,
   double spacing_x,
   double spacing_y,
   double spacing_z,
   const double *direction, // 9 doubles row-major, or NULL for identity
   int gzip_level
) {
   memset(w, 0, sizeof(*w));
   w->nsteps  = 0;
   w->npoints = nx * ny * nz;
   w->nx = nx;
   w->ny = ny;
   w->nz = nz;

   w->file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   if (w->file < 0) abort();

   w->root = H5Gcreate(w->file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (w->root < 0) abort();

   // ---- Root attributes (per VTKHDF 2.3 ImageData spec) ----
   {
      // Version = (2, 3)
      hsize_t vdims[1] = {2};
      hid_t vspace = H5Screate_simple(1, vdims, NULL);
      hid_t vattr = H5Acreate(w->root, "Version", H5T_NATIVE_INT, vspace,
                              H5P_DEFAULT, H5P_DEFAULT);
      int v[2] = {2, 3};
      vtkhdf__herr_ok(H5Awrite(vattr, H5T_NATIVE_INT, v), "H5Awrite Version");
      H5Aclose(vattr);
      H5Sclose(vspace);

      // Type = "ImageData"
      vtkhdf__write_string_attr(w->root, "Type", "ImageData");

      // Description (free-form, ignored by ParaView, useful for h5dump)
      vtkhdf__write_string_attr(w->root, "Description",
         "PFFDTD volumetric pressure snapshot (VTKHDF ImageData)");

      // WholeExtent = [0 nx-1 0 ny-1 0 nz-1]
      int64_t we[6] = {0, nx - 1, 0, ny - 1, 0, nz - 1};
      hsize_t edims[1] = {6};
      hid_t espace = H5Screate_simple(1, edims, NULL);
      hid_t eattr = H5Acreate(w->root, "WholeExtent", H5T_NATIVE_INT64, espace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(eattr, H5T_NATIVE_INT64, we), "H5Awrite WholeExtent");
      H5Aclose(eattr);
      H5Sclose(espace);

      // Origin, Spacing (3 doubles each)
      double origin[3]  = {origin_x,  origin_y,  origin_z};
      double spacing[3] = {spacing_x, spacing_y, spacing_z};
      hsize_t odims[1] = {3};
      hid_t ospace = H5Screate_simple(1, odims, NULL);
      hid_t oattr = H5Acreate(w->root, "Origin", H5T_NATIVE_DOUBLE, ospace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(oattr, H5T_NATIVE_DOUBLE, origin), "H5Awrite Origin");
      H5Aclose(oattr);
      hid_t sattr = H5Acreate(w->root, "Spacing", H5T_NATIVE_DOUBLE, ospace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(sattr, H5T_NATIVE_DOUBLE, spacing), "H5Awrite Spacing");
      H5Aclose(sattr);
      H5Sclose(ospace);

      // Direction = 3x3 matrix (row-major, 9 doubles)
      double dir[9];
      if (direction == NULL) {
         dir[0]=1; dir[1]=0; dir[2]=0;
         dir[3]=0; dir[4]=1; dir[5]=0;
         dir[6]=0; dir[7]=0; dir[8]=1;
      } else {
         memcpy(dir, direction, 9 * sizeof(double));
      }
      hsize_t ddims[1] = {9};
      hid_t dspace = H5Screate_simple(1, ddims, NULL);
      hid_t dattr = H5Acreate(w->root, "Direction", H5T_NATIVE_DOUBLE, dspace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(dattr, H5T_NATIVE_DOUBLE, dir), "H5Awrite Direction");
      H5Aclose(dattr);
      H5Sclose(dspace);
   }

   // ---- Steps group ----
   w->steps = H5Gcreate(w->root, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (w->steps < 0) abort();

   // Per-step time values (no compression, tiny dataset)
   w->ds_values = vtkhdf__create_1d_resizable(w->steps, "Values", H5T_IEEE_F32LE, 256, 0);

   // ---- PointData group ----
   w->pointData = H5Gcreate(w->root, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (w->pointData < 0) abort();

   // Pressure: float32, chunked one full snapshot per chunk so reads can map a step in
   // a single I/O.  Cap chunk size to ~1 GB worth of floats to stay within HDF5 limits.
   hsize_t chunk0 = (hsize_t)w->npoints;
   const hsize_t max_chunk_floats = (hsize_t)((1ULL << 30) / sizeof(float)); // 1 GB
   if (chunk0 > max_chunk_floats) chunk0 = max_chunk_floats;
   if (chunk0 < 1024) chunk0 = 1024;
   w->ds_pressure = vtkhdf__create_1d_resizable(w->pointData, "Pressure",
                                                H5T_IEEE_F32LE, chunk0, gzip_level);

   // Per-field offsets for time-dependent point data live under Steps/PointDataOffsets/<field>
   hid_t pdo = H5Gcreate(w->steps, "PointDataOffsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (pdo < 0) abort();
   w->ds_pressure_offsets = vtkhdf__create_1d_resizable(pdo, "Pressure",
                                                        H5T_NATIVE_INT64, 256, 0);
   H5Gclose(pdo);
}

// Append one time step.  pressure_f32 must have exactly npoints (= nx*ny*nz) values
// in the writer's flat layout (writer-X / nx fastest).
static inline void vtkhdf_image_writer_append_step_f32(
   VtkhdfImageWriter *w, float time_value, const float *pressure_f32, int64_t npoints
) {
   if (npoints != w->npoints) {
      fprintf(stderr, "VTKHDF writer error: npoints mismatch (got %ld, expected %ld)\n",
              (long)npoints, (long)w->npoints);
      abort();
   }

   // Record current offset in flattened Pressure dataset BEFORE appending
   hid_t fspace = H5Dget_space(w->ds_pressure);
   if (fspace < 0) abort();
   hsize_t cur_dims[1];
   H5Sget_simple_extent_dims(fspace, cur_dims, NULL);
   H5Sclose(fspace);
   int64_t start = (int64_t)cur_dims[0];

   vtkhdf__append_1d(w->ds_values,            H5T_IEEE_F32LE,    &time_value,   1);
   vtkhdf__append_1d(w->ds_pressure_offsets,  H5T_NATIVE_INT64,  &start,        1);
   vtkhdf__append_1d(w->ds_pressure,          H5T_IEEE_F32LE,    pressure_f32,  (hsize_t)npoints);

   w->nsteps += 1;
}

// Flush pending HDF5 buffers without closing.  Useful as a checkpoint during a long run.
static inline void vtkhdf_image_writer_flush(VtkhdfImageWriter *w) {
   if (w->file > 0) {
      vtkhdf__herr_ok(H5Fflush(w->file, H5F_SCOPE_GLOBAL), "H5Fflush");
   }
}

static inline void vtkhdf_image_writer_close(VtkhdfImageWriter *w) {
   // NSteps attribute on Steps group (ParaView reads this for the time slider)
   {
      hid_t as = H5Screate(H5S_SCALAR);
      hid_t a = H5Acreate(w->steps, "NSteps", H5T_NATIVE_INT64, as, H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(a, H5T_NATIVE_INT64, &w->nsteps), "H5Awrite NSteps");
      H5Aclose(a);
      H5Sclose(as);
   }

   if (w->ds_pressure_offsets > 0) H5Dclose(w->ds_pressure_offsets);
   if (w->ds_pressure         > 0) H5Dclose(w->ds_pressure);
   if (w->ds_values           > 0) H5Dclose(w->ds_values);

   if (w->pointData > 0) H5Gclose(w->pointData);
   if (w->steps     > 0) H5Gclose(w->steps);
   if (w->root      > 0) H5Gclose(w->root);
   if (w->file      > 0) H5Fclose(w->file);

   memset(w, 0, sizeof(*w));
}

#endif
