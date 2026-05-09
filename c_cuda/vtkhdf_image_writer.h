// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// VTKHDF ImageData writer (per-step file series + .pvd manifest)
//
// Writes a directory of self-contained STATIC VTKHDF 2.3 ImageData files plus
// a ParaView .pvd time-series manifest that ties them together.  This is the
// most reliable way to drive ParaView 5.12+ with transient ImageData: the
// single-file "transient ImageData" layout (4D PointData arrays prepended
// with a time axis + Steps/PointDataOffsets) is in the spec but is flaky in
// the released vtkHDFReader -- it loads without error but fails to display
// the field and segfaults on time advance.  The static reader is rock solid
// and a .pvd manifest gives the same time-slider UX with no perceivable
// per-step overhead for tens to a few hundred snapshots.
//
// File layout produced for `path = "/sim/run/vol_pressure.vtkhdf"`:
//
//     /sim/run/vol_pressure_0000.vtkhdf
//     /sim/run/vol_pressure_0001.vtkhdf
//     ...
//     /sim/run/vol_pressure_NNNN.vtkhdf
//     /sim/run/vol_pressure.pvd          <-- open THIS in ParaView
//
// Each .vtkhdf is a complete static ImageData with the same Origin /
// Spacing / Direction / WholeExtent and a single PointData/Pressure dataset
// of shape (nz, ny, nx) (writer-X / nx fastest, nz slowest), GZIP+SHUFFLE
// compressed at the requested level.
//
// Coordinate / ordering convention (unchanged from the prior single-file
// writer)
// ----------------------------------------------------------------------
// VTKHDF stores PointData arrays in (k, j, i) layout with i fastest.  In
// PFFDTD the engine grid is naturally (Nx_phys, Ny_phys, Nz_phys) with
// Nz_phys contiguous.  To get the volume sitting at the true physical
// (X, Y, Z) world coordinates of the room we map:
//
//     writer "X" axis (nx) <--> physical Z (fastest, contiguous)
//     writer "Y" axis (ny) <--> physical Y (middle)
//     writer "Z" axis (nz) <--> physical X (slowest)
//
// The caller passes a permutation Direction matrix
// {0,0,1, 0,1,0, 1,0,0} so ParaView interprets the writer-grid index
// (i, j, k) as world (xv[0]+k*h, yv[0]+j*h, zv[0]+i*h).  No host-side
// transpose needed -- the engine's flat buffer is written verbatim.
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
   // Per-step file naming
   char    base_path[1024];   //e.g. "/sim/run/vol_pressure"  (no extension)
   char    ext[32];           //e.g. ".vtkhdf"

   // Static geometry (written into every per-step file)
   double  origin[3];
   double  spacing[3];
   double  direction[9];
   int64_t nx, ny, nz;        //writer dims (nx fastest, nz slowest in flat layout)
   int64_t npoints;           //nx * ny * nz
   int     gzip_level;

   // Per-step time values, accumulated so we can emit the .pvd at close()
   float  *times;
   int64_t times_capacity;
   int64_t nsteps;
} VtkhdfImageWriter;

static inline void vtkhdf__herr_ok(herr_t status, const char *what) {
   if (status < 0) {
      fprintf(stderr, "VTKHDF writer error: %s\n", what);
      abort();
   }
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

// Split a path like "/foo/bar/vol_pressure.vtkhdf" into
//   base = "/foo/bar/vol_pressure"
//   ext  = ".vtkhdf"
// If no extension, ext defaults to ".vtkhdf".
// If path is too long it's truncated and a warning printed.
static inline void vtkhdf__split_path(const char *path,
                                      char *base, size_t base_sz,
                                      char *ext,  size_t ext_sz) {
   const size_t L = strlen(path);
   const char *dot = strrchr(path, '.');
   const char *slash = strrchr(path, '/');
   const char *bsl   = strrchr(path, '\\');
   const char *sep   = (slash > bsl) ? slash : bsl;

   if (dot != NULL && (sep == NULL || dot > sep)) {
      // ext starts at dot
      size_t base_len = (size_t)(dot - path);
      size_t ext_len  = L - base_len;
      if (base_len >= base_sz) base_len = base_sz - 1;
      if (ext_len  >= ext_sz)  ext_len  = ext_sz  - 1;
      memcpy(base, path, base_len); base[base_len] = '\0';
      memcpy(ext,  dot,  ext_len);  ext[ext_len]   = '\0';
   } else {
      // No extension: use the whole path as base, default ext
      size_t base_len = L;
      if (base_len >= base_sz) base_len = base_sz - 1;
      memcpy(base, path, base_len); base[base_len] = '\0';
      strncpy(ext, ".vtkhdf", ext_sz - 1); ext[ext_sz - 1] = '\0';
   }
}

// Open the writer.  This does NOT create any HDF5 file -- each step is its
// own self-contained file written by vtkhdf_image_writer_append_step_f32().
//
//   path         : output filename (use .vtkhdf extension for ParaView auto-detect).
//                  The writer derives a base + extension and writes
//                  <base>_<NNNN><ext> per step plus <base>.pvd.
//   nx, ny, nz   : writer-grid dimensions (nx fastest, nz slowest in flat layout)
//   origin_*     : world origin coords of voxel (0,0,0)
//   spacing_*    : world spacing of writer-grid axes
//   direction    : 9-element row-major 3x3, mapping (writer X, Y, Z) increments
//                  to world (X, Y, Z) increments.  Pass NULL for identity.
//   gzip_level   : 0 to disable, 1..9 for SHUFFLE+GZIP compression of the
//                  Pressure dataset in each per-step file.  3 is a good default.
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
   const double *direction,
   int gzip_level
) {
   memset(w, 0, sizeof(*w));

   vtkhdf__split_path(path, w->base_path, sizeof(w->base_path),
                            w->ext,       sizeof(w->ext));

   w->nx = nx;
   w->ny = ny;
   w->nz = nz;
   w->npoints = nx * ny * nz;
   w->gzip_level = gzip_level;

   w->origin[0]  = origin_x;
   w->origin[1]  = origin_y;
   w->origin[2]  = origin_z;
   w->spacing[0] = spacing_x;
   w->spacing[1] = spacing_y;
   w->spacing[2] = spacing_z;

   if (direction == NULL) {
      w->direction[0] = 1; w->direction[1] = 0; w->direction[2] = 0;
      w->direction[3] = 0; w->direction[4] = 1; w->direction[5] = 0;
      w->direction[6] = 0; w->direction[7] = 0; w->direction[8] = 1;
   } else {
      memcpy(w->direction, direction, 9 * sizeof(double));
   }

   // Initial allocation for time values; grows as needed.
   w->times_capacity = 64;
   w->times = (float *)malloc((size_t)w->times_capacity * sizeof(float));
   if (w->times == NULL) abort();
   w->nsteps = 0;

   printf("VTKHDF writer: file series at %s_<NNNN>%s plus %s.pvd\n",
          w->base_path, w->ext, w->base_path);
}

// Append one snapshot.  pressure_f32 must have exactly npoints (= nx*ny*nz)
// values in the writer's flat layout (nx fastest), interpreted as
// (nz, ny, nx) C-order when written as a 3D PointData/Pressure dataset.
static inline void vtkhdf_image_writer_append_step_f32(
   VtkhdfImageWriter *w, float time_value, const float *pressure_f32, int64_t npoints
) {
   if (npoints != w->npoints) {
      fprintf(stderr, "VTKHDF writer error: npoints mismatch (got %ld, expected %ld)\n",
              (long)npoints, (long)w->npoints);
      abort();
   }

   // Build per-step filename: "<base>_NNNN<ext>"
   char step_path[1280];
   int n = snprintf(step_path, sizeof(step_path), "%s_%04ld%s",
                    w->base_path, (long)w->nsteps, w->ext);
   if (n < 0 || n >= (int)sizeof(step_path)) {
      fprintf(stderr, "VTKHDF writer error: per-step path too long\n");
      abort();
   }

   hid_t file = H5Fcreate(step_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   if (file < 0) abort();

   hid_t root = H5Gcreate(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (root < 0) abort();

   // ---- Root attributes (per VTKHDF 2.3 ImageData spec) ----
   {
      // Version = (2, 3)
      hsize_t vdims[1] = {2};
      hid_t vspace = H5Screate_simple(1, vdims, NULL);
      hid_t vattr = H5Acreate(root, "Version", H5T_NATIVE_INT, vspace,
                              H5P_DEFAULT, H5P_DEFAULT);
      int v[2] = {2, 3};
      vtkhdf__herr_ok(H5Awrite(vattr, H5T_NATIVE_INT, v), "H5Awrite Version");
      H5Aclose(vattr);
      H5Sclose(vspace);

      vtkhdf__write_string_attr(root, "Type", "ImageData");
      vtkhdf__write_string_attr(root, "Description",
         "PFFDTD volumetric pressure snapshot (VTKHDF ImageData)");

      // WholeExtent = [0 nx-1 0 ny-1 0 nz-1]
      int64_t we[6] = {0, w->nx - 1, 0, w->ny - 1, 0, w->nz - 1};
      hsize_t edims[1] = {6};
      hid_t espace = H5Screate_simple(1, edims, NULL);
      hid_t eattr = H5Acreate(root, "WholeExtent", H5T_NATIVE_INT64, espace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(eattr, H5T_NATIVE_INT64, we), "H5Awrite WholeExtent");
      H5Aclose(eattr);
      H5Sclose(espace);

      // Origin, Spacing (3 doubles each), Direction (9 doubles)
      hsize_t odims[1] = {3};
      hid_t ospace = H5Screate_simple(1, odims, NULL);
      hid_t oattr = H5Acreate(root, "Origin", H5T_NATIVE_DOUBLE, ospace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(oattr, H5T_NATIVE_DOUBLE, w->origin), "H5Awrite Origin");
      H5Aclose(oattr);
      hid_t sattr = H5Acreate(root, "Spacing", H5T_NATIVE_DOUBLE, ospace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(sattr, H5T_NATIVE_DOUBLE, w->spacing), "H5Awrite Spacing");
      H5Aclose(sattr);
      H5Sclose(ospace);

      hsize_t ddims[1] = {9};
      hid_t dspace = H5Screate_simple(1, ddims, NULL);
      hid_t dattr = H5Acreate(root, "Direction", H5T_NATIVE_DOUBLE, dspace,
                              H5P_DEFAULT, H5P_DEFAULT);
      vtkhdf__herr_ok(H5Awrite(dattr, H5T_NATIVE_DOUBLE, w->direction), "H5Awrite Direction");
      H5Aclose(dattr);
      H5Sclose(dspace);
   }

   // ---- PointData/Pressure: 3D (nz, ny, nx) static dataset ----
   hid_t pointData = H5Gcreate(root, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (pointData < 0) abort();

   hsize_t pdims[3] = {(hsize_t)w->nz, (hsize_t)w->ny, (hsize_t)w->nx};
   hid_t pspace = H5Screate_simple(3, pdims, NULL);
   if (pspace < 0) abort();

   hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
   if (dcpl < 0) abort();
   if (w->gzip_level > 0) {
      // Chunk one full snapshot per chunk so this read/write is one I/O.
      // HDF5's hard chunk limit is 4 GB; our float32 snapshot fits safely
      // (1 billion points = 4 GB).  Apply SHUFFLE + GZIP.
      hsize_t chunk[3] = {(hsize_t)w->nz, (hsize_t)w->ny, (hsize_t)w->nx};
      vtkhdf__herr_ok(H5Pset_chunk(dcpl, 3, chunk), "H5Pset_chunk");
      vtkhdf__herr_ok(H5Pset_shuffle(dcpl), "H5Pset_shuffle");
      vtkhdf__herr_ok(H5Pset_deflate(dcpl, (unsigned)w->gzip_level), "H5Pset_deflate");
   }

   hid_t ds = H5Dcreate(pointData, "Pressure", H5T_IEEE_F32LE, pspace,
                        H5P_DEFAULT, dcpl, H5P_DEFAULT);
   if (ds < 0) abort();
   vtkhdf__herr_ok(H5Dwrite(ds, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, pressure_f32), "H5Dwrite Pressure");
   H5Dclose(ds);
   H5Pclose(dcpl);
   H5Sclose(pspace);

   H5Gclose(pointData);
   H5Gclose(root);
   H5Fclose(file);

   // ---- Record time for the .pvd manifest ----
   if (w->nsteps >= w->times_capacity) {
      w->times_capacity *= 2;
      float *grown = (float *)realloc(w->times, (size_t)w->times_capacity * sizeof(float));
      if (grown == NULL) abort();
      w->times = grown;
   }
   w->times[w->nsteps] = time_value;
   w->nsteps += 1;
}

// Flush is a no-op now (each step is closed immediately), retained for API
// compatibility with the previous single-file writer.
static inline void vtkhdf_image_writer_flush(VtkhdfImageWriter *w) {
   (void)w;
}

// Helper: extract just the basename portion of a path
static inline const char *vtkhdf__basename(const char *path) {
   const char *slash = strrchr(path, '/');
   const char *bsl   = strrchr(path, '\\');
   const char *sep   = (slash > bsl) ? slash : bsl;
   return (sep == NULL) ? path : (sep + 1);
}

// Close the writer: emit the .pvd manifest and free per-step bookkeeping.
static inline void vtkhdf_image_writer_close(VtkhdfImageWriter *w) {
   // .pvd path: "<base>.pvd"
   char pvd_path[1080];
   int n = snprintf(pvd_path, sizeof(pvd_path), "%s.pvd", w->base_path);
   if (n < 0 || n >= (int)sizeof(pvd_path)) {
      fprintf(stderr, "VTKHDF writer error: .pvd path too long\n");
      abort();
   }

   FILE *fp = fopen(pvd_path, "wb");
   if (fp == NULL) {
      fprintf(stderr, "VTKHDF writer error: cannot open %s for writing\n", pvd_path);
      abort();
   }

   const char *base_name = vtkhdf__basename(w->base_path);

   fprintf(fp, "<?xml version=\"1.0\"?>\n");
   fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
   fprintf(fp, "  <Collection>\n");
   for (int64_t i = 0; i < w->nsteps; i++) {
      // Reference per-step files by RELATIVE name so the directory is portable
      fprintf(fp,
         "    <DataSet timestep=\"%.9g\" group=\"\" part=\"0\" "
         "file=\"%s_%04ld%s\"/>\n",
         (double)w->times[i], base_name, (long)i, w->ext);
   }
   fprintf(fp, "  </Collection>\n");
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);

   printf("VTKHDF writer: %ld steps written, manifest at %s\n",
          (long)w->nsteps, pvd_path);

   if (w->times != NULL) free(w->times);
   memset(w, 0, sizeof(*w));
}

#endif
