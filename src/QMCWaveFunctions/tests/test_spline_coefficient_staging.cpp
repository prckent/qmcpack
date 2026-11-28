//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_template_test_macros.hpp>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "Concurrency/UtilityFunctions.hpp"
#include "Message/CommOperators.h"
#include "Message/Communicate.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/BsplineFactory/OneSplineOrbData.hpp"
#include "QMCWaveFunctions/BsplineFactory/SplineSetReader.h"
#include "io/hdf/hdf_archive.h"
#include "spline2/MultiBspline.hpp"
#if defined(HAVE_MPI)
#include "spline2/MultiBsplineMPIShared.hpp"
#endif

namespace qmcplusplus
{
namespace
{
class StagingInputFile
{
  std::filesystem::path directory_;

public:
  StagingInputFile()
  {
    // Each MPI rank writes its own fixture, even when tests share a working directory.
    auto pattern = (std::filesystem::temp_directory_path() / "qmcpack-spline-staging-XXXXXX").string();
    if (!::mkdtemp(pattern.data()))
      throw std::runtime_error("Could not create a temporary directory for the spline staging test");
    directory_ = pattern;
  }

  ~StagingInputFile()
  {
    std::error_code ignored;
    std::filesystem::remove(path(), ignored);
    std::filesystem::remove(directory_, ignored);
  }

  std::filesystem::path path() const { return directory_ / "orbitals.h5"; }
};

Vector<std::complex<double>> planeWaveCoefficients(int band)
{
  Vector<std::complex<double>> coefs(2);
  coefs[0] = std::polar(0.8, 0.01 * (band + 1));
  coefs[1] = std::polar(0.6, 0.02 * (band + 1));
  return coefs;
}

template<typename ST>
void check_staged_coefficients(bool duplex, int distributed_ranks, bool expect_staging = true)
{
  // The real-spline FFT postprocessing uses a reduction; fix its order for exact comparison.
  const Concurrency::OverrideMaxCapacity<Concurrency::Executor::OPENMP> single_thread(1);
  auto& world = *OHMMS::Controller;
  const int slots_per_orbital = duplex ? 2 : 1;
  constexpr int orbitals_per_block = 64;
  const int leaders_per_block = distributed_ranks == 1 ? 1 : world.size() / distributed_ranks;
  // Give each leader in the last block either two full batches and a tail, or a range too short to stage.
  const int orbitals_per_last_block_leader = expect_staging ? 17 : 7;
  const int norbitals = (distributed_ranks - 1) * orbitals_per_block + orbitals_per_last_block_leader * leaders_per_block;
  const TinyVector<int, 3> mesh{5, 6, 7};
  const TinyVector<int, 3> half_g{0, 0, 0};
  const TinyVector<double, 3> twist{0.0, 0.0, 0.0};
  const std::vector<TinyVector<int, 3>> gvecs{{0, 0, 0}, {1, -1, 2}};
  Ugrid grid[3]{};
  BCtype_d bc[3]{};
  for (int d = 0; d < 3; ++d)
  {
    grid[d].end = 1.0;
    grid[d].num = mesh[d];
    bc[d].lCode = bc[d].rCode = PERIODIC;
  }

  std::optional<StagingInputFile> input;
  BandInfoGroup bands;
  bands.myBands.resize(norbitals);
  bands.NumSPOs = norbitals;
  aligned_vector<int> band_map(norbitals);
  std::filesystem::path input_path;
  int fixture_failures = 0;
  try
  {
    input.emplace();
    input_path = input->path();
    hdf_archive h5;
    if (!h5.create(input_path))
      throw std::runtime_error("Could not create " + input_path.string());
    for (const std::string group : {"electrons", "kpoint_0", "spin_0"})
      h5.push(group);
    for (int band = 0; band < norbitals; ++band)
    {
      bands.myBands[band].BandIndex = band;
      band_map[band] = norbitals - 1 - band;
      h5.push("state_" + std::to_string(band));
      h5.write(planeWaveCoefficients(band), "psi_g");
      h5.pop();
    }
  }
  catch (const std::exception& error)
  {
    fixture_failures = 1;
    WARN("Rank " << world.rank() << " could not write the spline fixture: " << error.what());
  }
  world.allreduce(fixture_failures);
  REQUIRE(fixture_failures == 0);

  SimulationCell cell;
  ParticleSet electrons(cell);
  electrons.create({1});
  EinsplineSetBuilder::PSetMap psets;
  EinsplineSetBuilder builder(electrons, psets, world, nullptr);
  builder.MeshSize = mesh;
  builder.primcell_kpoints = {twist};
  builder.Gvecs = {gvecs};
  builder.H5FileName = input_path;
  SplineSetReader<ST> reader(&builder, duplex);
  reader.setRotate(false);

  auto dist_comm = world.split(world.size() / distributed_ranks);
  const int block_size = orbitals_per_block * slots_per_orbital;
  std::unique_ptr<MultiBsplineBase<ST>> actual;
#if defined(HAVE_MPI)
  if (distributed_ranks > 1)
    actual = std::make_unique<MultiBsplineMPIShared<ST>>(
        grid, bc, distributed_ranks * block_size,
        std::make_unique<Communicate>(world.split(world.size() / distributed_ranks)), distributed_ranks);
  else
#endif
    actual = std::make_unique<MultiBspline<ST>>(grid, bc, block_size);

  const int iblock = dist_comm.rank();
  const auto& offsets = actual->getBlockOffsets();
  const int first = offsets[iblock] / slots_per_orbital;
  const int last = std::min(norbitals, static_cast<int>(offsets[iblock + 1] / slots_per_orbital));
  auto& block = actual->getBlock(iblock);
  constexpr ST untouched = -123;
  std::fill_n(block.coefs, block.coefs_size, untouched);
  const int staged = reader.initialize_spline_pio_gather(0, bands, half_g, band_map, *actual, dist_comm);
  CAPTURE(duplex, world.rank(), iblock, first, last);
  if (iblock == distributed_ranks - 1)
    CHECK(staged == (expect_staging ? orbitals_per_last_block_leader : 0));
  else
    CHECK(staged >= 8);

  // Only the gather root has every orbital in this block.
  if (dist_comm.getInterGroupComm().rank() != 0)
    return;

  MultiBspline<ST> expected(grid, bc, offsets[iblock + 1] - offsets[iblock]);
  auto& reference = expected.getBlock(0);
  std::fill_n(reference.coefs, reference.coefs_size, untouched);
  OneSplineOrbData orbital(mesh, half_g, duplex);
  for (int iorb = first; iorb < last; ++iorb)
  {
    orbital.fft_spline(planeWaveCoefficients(band_map[iorb]), gvecs, twist, false);
    const int slot = (iorb - first) * slots_per_orbital;
    expected.setOneSpline(orbital.get_spline_r(), slot);
    if (duplex)
      expected.setOneSpline(orbital.get_spline_i(), slot + 1);
  }
  CHECK(block.coefs_size == reference.coefs_size);
  if (block.coefs_size != reference.coefs_size)
    return;
  auto* const end = block.coefs + block.coefs_size;
  const auto mismatch = std::mismatch(block.coefs, end, reference.coefs);
  if (mismatch.first != end)
  {
    const auto index = mismatch.first - block.coefs;
    const auto point = index / block.z_stride;
    const auto slot  = index % block.z_stride;
    CAPTURE(point, slot);
    CHECK(*mismatch.first == *mismatch.second);
  }
}
} // namespace

#if defined(QMC_MIXED_PRECISION)
#define STAGING_TEST_TYPES float
#else
#define STAGING_TEST_TYPES float, double
#endif

TEMPLATE_TEST_CASE("Spline reader stages coefficients without changing them", "[wavefunction][spline_staging]",
                   STAGING_TEST_TYPES)
{
  if (OHMMS::Controller->size() != 1)
    return;
  check_staged_coefficients<TestType>(false, 1);
  check_staged_coefficients<TestType>(true, 1);
}

TEMPLATE_TEST_CASE("Spline reader inserts short orbital ranges without staging", "[wavefunction][spline_staging]",
                   STAGING_TEST_TYPES)
{
  if (OHMMS::Controller->size() != 1)
    return;
  check_staged_coefficients<TestType>(false, 1, false);
}

#if defined(HAVE_MPI)
TEMPLATE_TEST_CASE("Spline reader gathers staged coefficients over distributed blocks", "[wavefunction][spline_staging]",
                   STAGING_TEST_TYPES)
{
  if (OHMMS::Controller->size() != 6)
    return;
  check_staged_coefficients<TestType>(false, 2);
  check_staged_coefficients<TestType>(true, 2);
}
#endif

#undef STAGING_TEST_TYPES
} // namespace qmcplusplus
