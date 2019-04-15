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


#include "QMCDrivers/DriftModifiers/DriftModifierBuilder.h"
#include "QMCDrivers/DriftModifiers/DriftModifierUNR.h"
#include "OhmmsData/ParameterSet.h"

namespace qmcplusplus
{
DriftModifierBase* createDriftModifier(xmlNodePtr cur)
{
  DriftModifierBase *DriftModifier;
  std::string ModifierName("UNR");
  ParameterSet m_param;
  m_param.add(ModifierName, "drift_modifier", "string");
  m_param.put(cur);
  if (ModifierName=="UNR")
    DriftModifier = new DriftModifierUNR;
  else
    APP_ABORT("createDriftModifier unknown drift_modifier " + ModifierName);
  DriftModifier->parseXML(cur);
  return DriftModifier;
}

} // namespace qmcplusplus
