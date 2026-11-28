//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2023 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@inte.com, Intel Corp.
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "SplineSetReader.h"
#include <algorithm>
#include "OneSplineOrbData.hpp"
#include "Utilities/FairDivide.h"
#include <Timer.h>
#if defined(QMC_COMPLEX)
#include "SplineC2C.h"
#else
#include "SplineR2R.h"
#include "SplineC2R.h"
#endif
#include "Message/CommOperators.h"
#include "spline2/SplineUtils.h"
#include "spline2/MultiBspline.hpp"
#include "spline2/MultiBsplineOffloadMapper.hpp"
#if defined(HAVE_MPI)
#include "spline2/MultiBsplineMPIShared.hpp"
#endif
#if defined(QMC_CCL)
#include "spline2/MultiBsplineOffloadMapperCCLShared.hpp"
#endif


namespace qmcplusplus
{
template<typename ST>
SplineSetReader<ST>::SplineSetReader(EinsplineSetBuilder* e, bool use_duplex_splines)
    : BsplineReader(e, use_duplex_splines)
{}

template<typename ST>
std::unique_ptr<SPOSet> SplineSetReader<ST>::create_spline_set(const std::string& my_name,
                                                               int spin,
                                                               const std::pair<int, int>& distributed_and_shared_ranks,
                                                               const BandInfoGroup& bandgroup)
{
  if (use_offload)
    app_summary() << "    Running OpenMP offload code path." << std::endl;
  else
    app_summary() << "    Running on CPU." << std::endl;

  auto [distributed_ranks, shared_ranks] = distributed_and_shared_ranks;

#if !defined(QMC_CCL)
  if (use_offload)
  {
    if (distributed_ranks > 1)
      app_warning() << "Offload implemenation doesn't support distributing the memory of spline "
                       "coefficients. Overriding distributed_ranks to 1."
                    << std::endl;
    distributed_ranks = 1;
  }
#endif

  auto dist_comm_ptr =
      std::make_unique<Communicate>(myComm.split(myComm.size() / (distributed_ranks * shared_ranks)));

  app_log() << "  Using " << (use_duplex_splines_ ? "complex" : "real") << " einspline table." << std::endl;
  if (distributed_ranks > 1)
    app_log() << "  Distributed across " << distributed_ranks << " MPI ranks." << std::endl;
  if (shared_ranks > 1)
    app_log() << "  Shared across " << shared_ranks << " MPI ranks." << std::endl;

  const TinyVector<int, 3> half_g = use_duplex_splines_
      ? TinyVector<int, 3>(0, 0, 0)
      : computeHalfG(mybuilder->TargetPtcl.getLattice().BoxBConds, mybuilder->primcell_kpoints,
                     bandgroup.myBands[0].TwistIndex);

  Ugrid xyz_grid[3];
  typename bspline_traits<ST, 3>::BCType xyz_bc[3];
  set_grid(mybuilder->MeshSize, half_g, xyz_grid, xyz_bc);

  const int N              = bandgroup.getNumDistinctOrbitals();
  const size_t num_splines = getAlignedSize<ST>(use_duplex_splines_ ? N * 2 : N);
  std::unique_ptr<MultiBsplineBase<ST>> multi_splines_ptr;
  std::unique_ptr<MultiBsplineOffloadMapperBase<ST>> offload_mapper;
#if defined(HAVE_MPI)
  if (distributed_ranks * shared_ranks > 1)
  {
    auto mpi_shared_splines = std::make_unique<MultiBsplineMPIShared<ST>>(xyz_grid, xyz_bc, num_splines,
                                                                          std::move(dist_comm_ptr), distributed_ranks);
#if defined(QMC_CCL)
    if (use_offload && distributed_ranks > 1)
      offload_mapper = std::make_unique<MultiBsplineOffloadMapperCCLShared<ST>>(*mpi_shared_splines);
#endif
    multi_splines_ptr = std::move(mpi_shared_splines);
  }
  else
#endif
    multi_splines_ptr = std::make_unique<MultiBspline<ST>>(xyz_grid, xyz_bc, num_splines);

  auto& multi_splines(*multi_splines_ptr);
  app_log() << "MEMORY " << multi_splines.sizeInByte() / (1 << 20) << " MB allocated "
            << "for the coefficients in 3D spline orbital representation" << std::endl;

  if (use_offload && !offload_mapper)
    offload_mapper = std::make_unique<MultiBsplineOffloadMapper<ST>>(*multi_splines_ptr);

  std::unique_ptr<BsplineSet> bspline;
#if defined(QMC_COMPLEX)
  bspline = std::make_unique<SplineC2C<ST>>(my_name, bandgroup.getNumSPOs(), mybuilder->PrimCell,
                                            std::move(multi_splines_ptr), std::move(offload_mapper));
#else
  if (use_duplex_splines_)
    bspline = std::make_unique<SplineC2R<ST>>(my_name, bandgroup.getNumSPOs(), mybuilder->PrimCell,
                                              std::move(multi_splines_ptr), std::move(offload_mapper));
  else
    bspline = std::make_unique<SplineR2R<ST>>(my_name, bandgroup.getNumSPOs(), mybuilder->PrimCell,
                                              std::move(multi_splines_ptr), std::move(offload_mapper));
#endif

  app_log() << "  ClassName = " << bspline->getClassName() << std::endl;

  bspline->resizeStorage(N);

  bspline->HalfG = half_g;

  if (use_duplex_splines_)
  {
    //baseclass handles twists
    check_twists(*bspline, bandgroup);
  }

  bool foundspline = lookforSplineDataDumpFile(bandgroup, bspline->getKeyword(), sizeof(ST));
  if (foundspline && myComm.rank() == 0)
  {
    Timer now;
    hdf_archive h5f;
    const auto splinefile = getSplineDumpFileName(bandgroup);
    h5f.open(splinefile, H5F_ACC_RDONLY);
    foundspline = SplineUtils<ST>::read(multi_splines, h5f);
    if (foundspline)
      app_log() << "  Successfully restored 3D B-spline coefficients from " << splinefile << ". The reading time is "
                << now.elapsed() << " sec." << std::endl;
  }

  /* create a sub communicator. spline table is shared across MPI ranks with identical subcomm rank id.
   *  myComm.size() == 12 ; shared_ranks = 2; distributed_ranks = 3;
   *          | group 0 | group 1 |
   *          |  0 1 2  |  3 4 5  |
   *          |  6 7 8  | 9 10 11 |
   */
  Communicate shared_comm(myComm.splitByStride(shared_ranks, distributed_ranks));
  Communicate dist_comm(shared_comm.split(shared_comm.size() / distributed_ranks));

  if (!foundspline)
  {
    if (shared_comm.getGroupID() == 0)
    {
      multi_splines.flush_zero(dist_comm.rank());
      Timer now;
      initialize_spline_pio_gather(spin, bandgroup, half_g, bspline->BandIndexMap, multi_splines, dist_comm);
      app_log() << "  SplineSetReader initialize_spline_pio " << now.elapsed() << " sec" << std::endl;
    }

    if (saveSplineCoefs && myComm.rank() == 0)
    {
      Timer now;
      const std::string splinefile(getSplineDumpFileName(bandgroup));
      hdf_archive h5f;
      h5f.create(splinefile);
      std::string classname = bspline->getClassName();
      h5f.write(classname, "class_name");
      int sizeD = sizeof(ST);
      h5f.write(sizeD, "sizeof");
      SplineUtils<ST>::write(multi_splines, h5f);
      h5f.close();
      app_log() << "  Stored spline coefficients in " << splinefile << " for potential reuse. The writing time is "
                << now.elapsed() << " sec." << std::endl;
    }
  }

  {
    myComm.barrier();
    Timer now;
    if (shared_comm.getGroupID() == 0)
      SplineUtils<ST>::bcast(multi_splines, dist_comm.rank(), dist_comm.getInterGroupComm());
    myComm.barrier();
    app_log() << "  Time to bcast the table = " << now.elapsed() << std::endl;
  }

  return bspline;
}

template<typename ST>
int SplineSetReader<ST>::initialize_spline_pio_gather(const int spin,
                                                      const BandInfoGroup& bandgroup,
                                                      const TinyVector<int, 3>& half_g,
                                                      const aligned_vector<int>& band_index_map,
                                                      MultiBsplineBase<ST>& multi_splines,
                                                      Communicate& dist_comm) const
{
  const auto& block_offsets = multi_splines.getBlockOffsets();
  const size_t iblock       = dist_comm.rank();
  auto& comm                = dist_comm.getInterGroupComm();
  //distribute bands over processor groups
  const int orb_block_start = use_duplex_splines_ ? block_offsets[iblock] / 2 : block_offsets[iblock];
  const int orb_block_end   = use_duplex_splines_ ? block_offsets[iblock + 1] / 2 : block_offsets[iblock + 1];
  const int Nbands          = std::min(orb_block_end, bandgroup.getNumDistinctOrbitals()) - orb_block_start;
  const int Nprocs          = comm.size();
  const int Nbandgroups     = Nbands > 0 ? std::min(Nbands, Nprocs) : 1;

  Communicate band_group_comm(comm.split(Nbandgroups));
  std::vector<int> band_groups(Nbandgroups + 1, 0);
  FairDivideLow(Nbands, Nbandgroups, band_groups);
  const int iorb_first = orb_block_start + band_groups[band_group_comm.getGroupID()];
  const int iorb_last  = orb_block_start + band_groups[band_group_comm.getGroupID() + 1];

  app_log() << "Start transforming plane waves to 3D B-Splines." << std::endl;
  OneSplineOrbData oneband(mybuilder->MeshSize, half_g, use_duplex_splines_);
  Vector<std::complex<double>> cG(mybuilder->Gvecs[0].size());
  const std::vector<BandInfo>& cur_bands = bandgroup.myBands;
  int staged_orbitals = 0;
  if (band_group_comm.isGroupLeader())
  {
    auto& group_leader_comm = band_group_comm.getInterGroupComm();
    hdf_archive h5f;
    h5f.open(mybuilder->H5FileName, H5F_ACC_RDONLY);
    auto& block                        = multi_splines.getBlock(iblock);
    const int splines_per_orbital       = use_duplex_splines_ ? 2 : 1;
    constexpr int batch_size           = 8;
    constexpr size_t max_staging_bytes = 64 * 1024 * 1024; // Per leader.
    const int batch_splines            = batch_size * splines_per_orbital;

    // Match setOneSpline's grid traversal, including the three boundary points per axis,
    // but require consecutive grid points in both tables for the flat copy.
    const intptr_t nx   = block.x_grid.num + 3;
    const intptr_t ny   = block.y_grid.num + 3;
    const intptr_t nz   = block.z_grid.num + 3;
    const size_t points = static_cast<size_t>(nx) * ny * nz;
    const auto is_contiguous_source = [&](const UBspline_3d_d& spline) {
      return spline.x_grid.num == block.x_grid.num && spline.y_grid.num == block.y_grid.num &&
          spline.z_grid.num == block.z_grid.num && spline.x_stride == ny * nz && spline.y_stride == nz &&
          spline.coefs_size == points;
    };
    const bool stage_coefficients = iorb_last - iorb_first >= batch_size && block.z_stride >= 64 &&
        points <= max_staging_bytes / (batch_splines * sizeof(ST)) &&
        block.x_stride == ny * nz * block.z_stride && block.y_stride == nz * block.z_stride &&
        block.coefs_size == points * block.z_stride && is_contiguous_source(oneband.get_spline_r()) &&
        (!use_duplex_splines_ || is_contiguous_source(oneband.get_spline_i()));
    // Stage one contiguous grid per spline in destination precision. On flush,
    // transpose the batch into the point-major table: writing adjacent spline
    // slots together uses more of each destination cache line than writing an
    // entire spline with a large stride.
    std::vector<ST> staging(stage_coefficients ? points * batch_splines : 0);
    // Copy the freshly computed spline into its spline slot in the batch
    // before the next orbital overwrites the single-spline workspace.
    auto stage = [&](const UBspline_3d_d& spline, int slot) {
      std::copy_n(spline.coefs, points, staging.data() + points * slot);
    };
    for (int iorb = iorb_first; iorb < iorb_last; iorb++)
    {
      const auto& cur_band = cur_bands[band_index_map[iorb]];
      const int ti         = cur_band.TwistIndex;
      readOneOrbitalCoefs(psi_g_path(ti, spin, cur_band.BandIndex), h5f, cG);
      oneband.fft_spline(cG, mybuilder->Gvecs[0], mybuilder->primcell_kpoints[ti], rotate);
      if (stage_coefficients)
      {
        const int slot = (iorb - iorb_first) % batch_size; // Orbital's position in the current batch.
        stage(oneband.get_spline_r(), slot * splines_per_orbital);
        if (use_duplex_splines_)
          stage(oneband.get_spline_i(), slot * splines_per_orbital + 1);

        // Flush full batches and any partial batch at the end of this leader's range.
        if (slot + 1 == batch_size || iorb + 1 == iorb_last)
        {
          // Convert the first orbital in the batch to a block-local spline index.
          const size_t start = (iorb - slot) * splines_per_orbital - block_offsets[iblock];
          const int count    = (slot + 1) * splines_per_orbital; // Filled spline slots, excluding padding.
          // point indexes the flattened grid, and i indexes a staged spline.
          for (size_t point = 0; point < points; ++point)
          {
            // First destination slot for this batch at the current grid location.
            ST* out = block.coefs + point * block.z_stride + start;
            for (int i = 0; i < count; ++i)
              out[i] = staging[i * points + point];
          }
          staged_orbitals += slot + 1;
        }
      }
      else if (use_duplex_splines_)
      {
        multi_splines.setOneSpline(oneband.get_spline_r(), iorb * 2);
        multi_splines.setOneSpline(oneband.get_spline_i(), iorb * 2 + 1);
      }
      else
        multi_splines.setOneSpline(oneband.get_spline_r(), iorb);
    }

    {
      group_leader_comm.barrier();
      Timer now;
      if (use_duplex_splines_)
      {
        std::vector<int> offset(band_groups.size());
        for (int i = 0; i < offset.size(); i++)
          offset[i] = band_groups[i] * 2;
        SplineUtils<ST>::gatherv(multi_splines, iblock, offset, group_leader_comm);
      }
      else
        SplineUtils<ST>::gatherv(multi_splines, iblock, band_groups, group_leader_comm);
      app_log() << "  Time to gather the table = " << now.elapsed() << std::endl;
    }
  }
  return staged_orbitals;
}

template class SplineSetReader<float>;
#if !defined(QMC_MIXED_PRECISION)
template class SplineSetReader<double>;
#endif
} // namespace qmcplusplus
