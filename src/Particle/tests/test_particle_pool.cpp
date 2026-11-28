//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by:  Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"


#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/InitMolecularSystem.h"
#include "Particle/ParticleSetPool.h"
#include "Utilities/RandomGenerator.h"


#include <stdio.h>
#include <string>
#include <sstream>
#include <type_traits>


namespace qmcplusplus
{
TEST_CASE("ParticleSetPool", "[qmcapp]")
{
  Communicate& c(*OHMMS::Controller);

  ParticleSetPool pp(c);

  // See ParticleIO/tests/test_xml_io.cpp for particle parsing
  const char* particles = R"(
<particleset name="ion0" size="1">
  <group name="He">
    <parameter name="charge">2</parameter>
  </group>
  <attrib name="position" datatype="posArray">
    0.1 0.2 0.3
  </attrib>
</particleset>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));

  xmlNodePtr root = doc.getRoot();

  pp.put(root);

  ParticleSet* ions = pp.getParticleSet("ion0");
  REQUIRE(ions != NULL);

  ParticleSet* not_here = pp.getParticleSet("does_not_exist");
  REQUIRE(not_here == NULL);

  MCWalkerConfiguration* ws = pp.getWalkerSet("ion0");
  REQUIRE(ws != NULL);

  REQUIRE_THROWS_AS(pp.getWalkerSet("does_not_exist"), std::runtime_error);

  auto ps2 = std::make_unique<ParticleSet>(pp.getSimulationCell());
  ps2->setName("particle_set_2");
  pp.addParticleSet(std::move(ps2));

  // should do nothing, since no random particlesets were specified
  pp.randomize();

  std::stringstream out;
  pp.get(out);
  //app_log() << "ParticleSetPool::get returns  " << std::endl;
  //app_log() << out.str() << std::endl;
}

TEST_CASE("ParticleSetPool random", "[qmcapp]")
{
  Communicate& c(*OHMMS::Controller);

  ParticleSetPool pp(c);

  // See ParticleIO/tests/test_xml_io.cpp for particle parsing
  const char* particles = R"(
<tmp>
<particleset name="ion0" size="1">
  <group name="He">
    <parameter name="charge">2</parameter>
  </group>
  <attrib name="position" datatype="posArray">
    0.1 0.2 0.3
  </attrib>
</particleset>
<particleset name="elec" random="yes" randomsrc="ion0" spinor="yes">
   <group name="u" size="4">
      <parameter name="charge">-1</parameter>
   </group>
</particleset>
</tmp>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));

  xmlNodePtr root = doc.getRoot();

  xmlNodePtr part_ion = xmlFirstElementChild(root);
  pp.put(part_ion);
  pp.put(xmlNextElementSibling(part_ion));

  ParticleSet* ions = pp.getParticleSet("ion0");
  REQUIRE(ions != NULL);
  REQUIRE(!ions->isSpinor());

  ParticleSet* elec = pp.getParticleSet("elec");
  REQUIRE(ions != NULL);
  REQUIRE(elec->isSpinor());
  REQUIRE(elec->R.size() == 4);
  REQUIRE(elec->spins.size() == 4);


  // should do something
  pp.randomize();

  REQUIRE(elec->R[0][0] != 0.0);
  REQUIRE(elec->spins[0] != 0.0);
}

TEST_CASE("Initialize electrons in the padded ion volume", "[qmcapp]")
{
  ParticleSetPool pool(*OHMMS::Controller);
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(R"(
<tmp>
  <simulationcell>
    <parameter name="lattice">20 0 0  0 20 0  0 0 20</parameter>
    <parameter name="bconds">p p p</parameter>
  </simulationcell>
  <particleset name="ion0">
    <group name="He" size="2">
      <parameter name="charge">2</parameter>
      <attrib name="position" datatype="posArray">6 6 6  14 14 14</attrib>
    </group>
  </particleset>
  <particleset name="elec">
    <group name="u" size="2">
      <parameter name="charge">-1</parameter>
    </group>
  </particleset>
  <init source="ion0" target="elec" use_volume="yes"/>
</tmp>
)"));
  xmlNodePtr cell      = xmlFirstElementChild(doc.getRoot());
  xmlNodePtr ions      = xmlNextElementSibling(cell);
  xmlNodePtr electrons = xmlNextElementSibling(ions);
  REQUIRE(pool.readSimulationCellXML(cell));
  REQUIRE(pool.put(ions));
  REQUIRE(pool.put(electrons));
  ParticleSet* elec = pool.getParticleSet("elec");
  REQUIRE(elec != nullptr);

  ParticleSet::PosType lower(4.0);
  ParticleSet::PosType upper(16.0);
  SECTION("Padding stays inside the cell") {}
  SECTION("Padding is clipped at the cell faces")
  {
    ParticleSet* ion_set = pool.getParticleSet("ion0");
    REQUIRE(ion_set != nullptr);
    ion_set->R[0] = ParticleSet::PosType(1.0, 5.0, 1.0);
    ion_set->R[1] = ParticleSet::PosType(11.0, 19.0, 19.0);
    lower         = ParticleSet::PosType(0.0, 3.0, 0.0);
    upper         = ParticleSet::PosType(13.0, 20.0, 20.0);
  }

  constexpr auto seed = 42;
  std::vector<RandomGenerator::uint_type> previous_state;
  Random.save(previous_state);
  Random.seed(seed);
  InitMolecularSystem initialize(pool);
  const bool initialized = initialize.put(xmlNextElementSibling(electrons));
  Random.load(previous_state);
  REQUIRE(initialized);
  REQUIRE(elec->R.size() == 2);

  RandomGenerator reference_rng(seed);
  for (size_t electron = 0; electron < elec->R.size(); ++electron)
    for (int dim = 0; dim < OHMMS_DIM; ++dim)
    {
      CAPTURE(electron, dim);
      CHECK(elec->R[electron][dim] == Approx(lower[dim] + (upper[dim] - lower[dim]) * reference_rng()));
    }
}

TEST_CASE("ParticleSetPool putLattice", "[qmcapp]")
{
  Communicate& c(*OHMMS::Controller);

  ParticleSetPool pp(c);

  const char* lattice = R"(<parameter name="lattice"> </parameter>)";

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(lattice));

  xmlNodePtr root = doc.getRoot();
  pp.readSimulationCellXML(root);
}
} // namespace qmcplusplus
