//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_DRIFTMODIFIER_BASE_H
#define QMCPLUSPLUS_DRIFTMODIFIER_BASE_H

#include "Configuration.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCHamiltonians/QMCHamiltonian.h"

namespace qmcplusplus
{
class DriftModifierBase
{
public:
  using RealType = QMCTraits::RealType;
  using PosType  = QMCTraits::PosType;
  using GradType = QMCTraits::GradType;

  /** evaluate a drift with a real force
   * @param tau timestep
   * @param qf quantum force
   * @param drift
   */
  virtual void getScaledDrift(const ParticleSet& P,
                              const TrialWaveFunction& W,
                              const QMCHamiltonian& H,
                              RealType tau,
                              GradType qf,
                              int iat,
                              PosType drift) const = 0;

  virtual bool parseXML(xmlNodePtr cur){};

protected:
  // modifer name
  std::string ClassName;
};

} // namespace qmcplusplus

#endif
