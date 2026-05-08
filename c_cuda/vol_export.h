// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// Volumetric snapshot export helpers for PFFDTD
//
// This header provides host-side routines that turn the engine's per-step
// pressure grid into a fully dense Cartesian volume suitable for VTKHDF
// ImageData export, then forwards it to vtkhdf_image_writer.
//
// Two cases are handled:
//   * Cartesian (fcc_flag == 0): the engine grid is already a dense Cartesian
//     volume of shape (Nx, Ny, Nz) with Nz contiguous.  No conversion needed --
//     the gathered host buffer is forwarded directly.
//
//   * FCC folded (fcc_flag == 2): the engine grid is shape (Nx, Ny, Nz) but
//     represents a doubly-compressed FCC sublattice:
//         - logical Y dimension is Nyf = 2*(Ny - 1)  (folded onto itself)
//         - only sites with even (ix + iy_logical + iz) carry data
//     We undo the fold via parity-based scatter, then "densify" the inactive
//     half-lattice by averaging the 12 nearest FCC neighbours (the stencil used
//     by the air kernel).  Result: a dense Cartesian (Nx, Nyf, Nz) volume of
//     the same physical spacing h, ready for FlyingEdges3D / contour /
//     volume rendering in ParaView.
///////////////////////////////////////////////////////////////////////////////

#ifndef _VOL_EXPORT_H
#define _VOL_EXPORT_H

#ifndef _STDINT_H
#include <stdint.h>
#endif
#ifndef _STDLIB_H
#include <stdlib.h>
#endif
#ifndef _STDIO_H
#include <stdio.h>
#endif
#ifndef _STRING_H
#include <string.h>
#endif
#ifndef _MATH_H
#include <math.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _FDTD_COMMON_H
#include <fdtd_common.h>  // for typedef Real
#endif
#include "vtkhdf_image_writer.h"

// ----------------------------------------------------------------------------
// VolExporter: small bundle that owns the writer and the host-side scratch
// buffers used per snapshot.  One instance per simulation run.
// ----------------------------------------------------------------------------
typedef struct VolExporter {
   bool enabled;
   int8_t fcc_flag;       // 0 = cartesian, 2 = FCC folded (1 = CCP, host-only path)
   int snapshot_stride;   // write a snapshot every N time steps (>=1)
   int gzip_level;        // GZIP level for Pressure dataset (0 disables)

   int64_t Nx;            // engine storage X (slowest)
   int64_t Ny;            // engine storage Y (folded for FCC)
   int64_t Nz;            // engine storage Z (contiguous)
   int64_t Nyf;           // logical/unfolded Y (= Ny for cart, 2*(Ny-1) for FCC)

   double h;              // grid spacing (metres)
   double origin_x;       // physical X origin (xv[0])
   double origin_y;       // physical Y origin (yv[0])
   double origin_z;       // physical Z origin (zv[0])

   // The writer is configured so that ParaView shows the volume in true
   // physical (X,Y,Z) world coordinates while we keep the engine's natural
   // Z-fast flat layout in the buffers we hand it.  See the long comment at
   // the top of vtkhdf_image_writer.h for the full derivation.
   VtkhdfImageWriter writer;

   // Host buffers
   float   *gather_buf;   // size Nx*Ny*Nz, gathered engine grid (Z-fast)
   float   *out_buf;      // size Nx*Nyf*Nz, dense Cartesian output (Z-fast)
                          //   for cart this aliases gather_buf
   double   rescale;      // multiply gathered samples by this on copy-in
                          //   (typically sd->infac after scale_input)
   int64_t  nsteps_written;
} VolExporter;

// ----------------------------------------------------------------------------
// Open the writer and allocate scratch buffers.  Should be called after the
// per-GPU malloc block and before the time loop.
//
//   path             : output file (.vtkhdf)
//   fcc_flag         : 0 (cartesian) or 2 (FCC folded)
//   Nx, Ny, Nz       : engine storage dims (Ny is folded for FCC)
//   xv0, yv0, zv0    : physical world coords of voxel (0,0,0)
//   h                : grid spacing in metres
//   stride           : write every `stride` time steps (must be >= 1)
//   gzip_level       : 0..9, recommend 3 for ~2x compression of pressure fields
//   rescale          : pass sd->infac to undo the input scaling, or 1.0 to
//                      record raw normalised values
//
// On disabled=true (set by caller before this), this function does nothing
// and the append/close calls below become no-ops.
// ----------------------------------------------------------------------------
static inline void vol_exporter_open(
   VolExporter *vx,
   const char *path,
   int8_t fcc_flag,
   int64_t Nx, int64_t Ny, int64_t Nz,
   double xv0, double yv0, double zv0,
   double h,
   int snapshot_stride,
   int gzip_level,
   double rescale
) {
   memset(vx, 0, sizeof(*vx));
   vx->enabled         = true;
   vx->fcc_flag        = fcc_flag;
   vx->snapshot_stride = (snapshot_stride > 0) ? snapshot_stride : 1;
   vx->gzip_level      = gzip_level;
   vx->Nx              = Nx;
   vx->Ny              = Ny;
   vx->Nz              = Nz;
   vx->Nyf             = (fcc_flag == 2) ? (2 * (Ny - 1)) : Ny;
   vx->h               = h;
   vx->origin_x        = xv0;
   vx->origin_y        = yv0;
   vx->origin_z        = zv0;
   vx->rescale         = rescale;

   // Permutation Direction so the writer's natural (Z-fastest) flat layout
   // displays in true physical (X, Y, Z) world coords.  See header comment in
   // vtkhdf_image_writer.h for the derivation.
   const double dir_perm_zyx_to_xyz[9] = {
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      1.0, 0.0, 0.0
   };

   // Writer dims: nx = Nz_phys (fastest), ny = Nyf_phys, nz = Nx_phys (slowest)
   vtkhdf_image_writer_open(
      &vx->writer,
      path,
      Nz,        // nx (writer)  = physical Nz
      vx->Nyf,   // ny (writer)  = physical Nyf (unfolded)
      Nx,        // nz (writer)  = physical Nx
      xv0, yv0, zv0,
      h, h, h,
      dir_perm_zyx_to_xyz,
      gzip_level
   );

   const size_t npts_storage = (size_t)Nx * (size_t)Ny * (size_t)Nz;
   const size_t npts_out     = (size_t)Nx * (size_t)vx->Nyf * (size_t)Nz;

   vx->gather_buf = (float *)malloc(npts_storage * sizeof(float));
   if (vx->gather_buf == NULL) {
      fprintf(stderr, "VolExporter: failed to allocate gather buffer (%.2f GB)\n",
              npts_storage * sizeof(float) / (1024.0 * 1024.0 * 1024.0));
      abort();
   }

   if (fcc_flag == 2) {
      vx->out_buf = (float *)malloc(npts_out * sizeof(float));
      if (vx->out_buf == NULL) {
         fprintf(stderr, "VolExporter: failed to allocate unfold buffer (%.2f GB)\n",
                 npts_out * sizeof(float) / (1024.0 * 1024.0 * 1024.0));
         abort();
      }
   } else {
      vx->out_buf = vx->gather_buf;  // alias, no extra memory
   }

   printf("\n");
   printf("========== VOL EXPORT ==========\n");
   printf("  path             : %s\n", path);
   printf("  fcc_flag         : %d\n", (int)fcc_flag);
   printf("  storage dims     : (Nx=%ld, Ny=%ld, Nz=%ld)\n", (long)Nx, (long)Ny, (long)Nz);
   printf("  output dims      : (Nx=%ld, Nyf=%ld, Nz=%ld)  -- physical XYZ\n",
          (long)Nx, (long)vx->Nyf, (long)Nz);
   printf("  origin (x,y,z)   : (%.6f, %.6f, %.6f) m\n", xv0, yv0, zv0);
   printf("  spacing h        : %.6f m\n", h);
   printf("  snapshot stride  : %d\n", vx->snapshot_stride);
   printf("  per-snapshot     : %.2f MB (output) %s\n",
          npts_out * sizeof(float) / (1024.0 * 1024.0),
          (gzip_level > 0) ? "uncompressed" : "");
   printf("  gzip level       : %d\n", gzip_level);
   printf("  rescale (infac)  : %.6e\n", rescale);
   printf("================================\n\n");
}

// ----------------------------------------------------------------------------
// Internal: unfold + densify for FCC (fcc_flag == 2)
//
// Storage layout (input, `gathered`):
//     shape (Nx, Ny, Nz), Nz contiguous
//     each cell carries one real FCC active value
//     parity rule: storage cell (ix, iy, iz) holds the value of
//        - logical (ix, iy,             iz) if (ix + iy + iz) % 2 == 0
//        - logical (ix, Nyf - 1 - iy,   iz) if (ix + iy + iz) % 2 == 1
//     (for iy in [1, Ny-2]; iy=0 / iy=Ny-1 are the fold halo rows)
//
// Output layout (`out`):
//     shape (Nx, Nyf, Nz), Nz contiguous
//     fully dense Cartesian; the logical FCC sublattice
//        ((ix + iy + iz) % 2 == 0)
//     is filled directly from the scatter; the other half is filled by
//     averaging the 12 nearest FCC neighbours (interior cells only).  Cells
//     missing neighbours (faces) get a partial average; if no neighbours are
//     available they remain at NaN (rare; only at extreme corners).
// ----------------------------------------------------------------------------
static inline void vol_export__unfold_and_densify_fcc(
   const float *gathered, float *out,
   int64_t Nx, int64_t Ny, int64_t Nz, int64_t Nyf
) {
   const int64_t NzNy_in  = Nz * Ny;
   const int64_t NzNyf_out = Nz * Nyf;

   // 1) Initialise output to NaN so any cell we never touch is obviously empty
   //    in ParaView (FlyingEdges3D / contour ignore NaN; volume rendering shows
   //    them as fully transparent).
   const float nanf = (float)NAN;
   const int64_t npts_out = Nx * Nyf * Nz;
   #pragma omp parallel for schedule(static)
   for (int64_t i = 0; i < npts_out; i++) out[i] = nanf;

   // 2) Scatter: storage cell -> logical Cartesian cell using parity rule.
   //    iy_storage in [0, Ny-1].  iy_storage = Ny-1 is the fold halo row that
   //    KernelFoldFCC mirrors from iy_storage = Ny-2 -- skip it (the data is a
   //    duplicate of (iy_storage = Ny-2) under the OPPOSITE parity, i.e. the
   //    logical (Nyf - (Ny-1) - 1 = Ny-2) which we already write below).
   //    iy_storage = 0 is the y=0 boundary; only one parity is meaningful.
   #pragma omp parallel for schedule(static) collapse(2)
   for (int64_t ix = 0; ix < Nx; ix++) {
      for (int64_t iy_s = 0; iy_s < Ny - 1; iy_s++) {
         const int64_t base_in  = ix * NzNy_in  + iy_s * Nz;
         for (int64_t iz = 0; iz < Nz; iz++) {
            const float v = gathered[base_in + iz];
            int64_t iy_log;
            if (((ix + iy_s + iz) & 1) == 0) {
               iy_log = iy_s;                  // lower half (or boundary y=0)
            } else {
               iy_log = Nyf - 1 - iy_s;        // upper half mirror
            }
            // skip writes that would land outside the unfolded volume (defensive)
            if (iy_log < 0 || iy_log >= Nyf) continue;
            out[ix * NzNyf_out + iy_log * Nz + iz] = v;
         }
      }
   }

   // 3) Densify: for every inactive cell ((ix + iy + iz) % 2 == 1) average
   //    available FCC nearest neighbours.  The 12 unit offsets in (dix, diy, diz):
   //
   //       (+/-1, +/-1,    0)   ->  4 entries
   //       (+/-1,    0, +/-1)   ->  4 entries
   //       (   0, +/-1, +/-1)   ->  4 entries
   //
   //    All 12 land on the active sublattice by parity, so we can read them
   //    directly from `out`.
   //
   //    Edge handling: skip neighbours that fall outside the volume; use a
   //    partial average over what's available.  If none are available the cell
   //    stays NaN (only at extreme corners with the "wrong" parity, vanishing
   //    visual impact).
   const int8_t off[12][3] = {
      {+1, +1,  0}, {+1, -1,  0}, {-1, +1,  0}, {-1, -1,  0},
      {+1,  0, +1}, {+1,  0, -1}, {-1,  0, +1}, {-1,  0, -1},
      { 0, +1, +1}, { 0, +1, -1}, { 0, -1, +1}, { 0, -1, -1},
   };

   #pragma omp parallel for schedule(static) collapse(2)
   for (int64_t ix = 0; ix < Nx; ix++) {
      for (int64_t iy = 0; iy < Nyf; iy++) {
         const int64_t base = ix * NzNyf_out + iy * Nz;
         const int parity_xy = (int)((ix + iy) & 1);
         for (int64_t iz = 0; iz < Nz; iz++) {
            if ((parity_xy ^ (int)(iz & 1)) == 0) continue;   // active site, skip

            float acc = 0.0f;
            int   cnt = 0;
            for (int n = 0; n < 12; n++) {
               int64_t jx = ix + off[n][0];
               int64_t jy = iy + off[n][1];
               int64_t jz = iz + off[n][2];
               if (jx < 0 || jx >= Nx)  continue;
               if (jy < 0 || jy >= Nyf) continue;
               if (jz < 0 || jz >= Nz)  continue;
               float vv = out[jx * NzNyf_out + jy * Nz + jz];
               if (vv != vv) continue;  // NaN guard (a neighbour was untouched)
               acc += vv;
               cnt++;
            }
            if (cnt > 0) {
               out[base + iz] = acc / (float)cnt;
            }
            // else: leave as NaN (unreachable at any well-formed snapshot)
         }
      }
   }
}

// ----------------------------------------------------------------------------
// Append a snapshot.  Caller is responsible for having gathered the engine
// pressure grid into vx->gather_buf BEFORE calling this (for cart it is the
// final destination, for FCC it is the staging area for unfold+densify).
//
//   time_value : world time in seconds for this snapshot
// ----------------------------------------------------------------------------
static inline void vol_exporter_append(VolExporter *vx, float time_value) {
   if (!vx->enabled) return;

   if (vx->rescale != 1.0) {
      const double r = vx->rescale;
      const int64_t n = vx->Nx * vx->Ny * vx->Nz;
      #pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < n; i++) {
         vx->gather_buf[i] = (float)(vx->gather_buf[i] * r);
      }
   }

   if (vx->fcc_flag == 2) {
      vol_export__unfold_and_densify_fcc(
         vx->gather_buf, vx->out_buf,
         vx->Nx, vx->Ny, vx->Nz, vx->Nyf
      );
   }
   // else: out_buf aliases gather_buf, nothing to do

   const int64_t npts_out = vx->Nx * vx->Nyf * vx->Nz;
   vtkhdf_image_writer_append_step_f32(&vx->writer, time_value, vx->out_buf, npts_out);
   vx->nsteps_written++;
}

// ----------------------------------------------------------------------------
// Close writer and free scratch buffers.
// ----------------------------------------------------------------------------
static inline void vol_exporter_close(VolExporter *vx) {
   if (!vx->enabled) return;
   vtkhdf_image_writer_close(&vx->writer);
   if (vx->out_buf != NULL && vx->out_buf != vx->gather_buf) {
      free(vx->out_buf);
   }
   if (vx->gather_buf != NULL) {
      free(vx->gather_buf);
   }
   printf("VOL EXPORT: wrote %ld snapshots\n", (long)vx->nsteps_written);
   memset(vx, 0, sizeof(*vx));
}

#endif // _VOL_EXPORT_H
