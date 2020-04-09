//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


/**@file DiracDeterminantBatched.h
 * @brief Declaration of DiracDeterminantBatched with a S(ingle)P(article)O(rbital)Set
 */
#ifndef QMCPLUSPLUS_DIRACDETERMINANTBATCHED_H
#define QMCPLUSPLUS_DIRACDETERMINANTBATCHED_H

#include "QMCWaveFunctions/Fermion/DiracDeterminantBase.h"
#include "QMCWaveFunctions/Fermion/DiracMatrix.h"

namespace qmcplusplus
{
template<typename DET_ENGINE_TYPE = DiracMatrix<QMCTraits::QTFull::ValueType>>
class DiracDeterminantBatched : public DiracDeterminantBase
{
public:
  using ValueVector_t = SPOSet::ValueVector_t;
  using ValueMatrix_t = SPOSet::ValueMatrix_t;
  using GradVector_t  = SPOSet::GradVector_t;
  using GradMatrix_t  = SPOSet::GradMatrix_t;
  using HessMatrix_t  = SPOSet::HessMatrix_t;
  using HessVector_t  = SPOSet::HessVector_t;
  using HessType      = SPOSet::HessType;

  using mValueType = QMCTraits::QTFull::ValueType;
  using mGradType  = TinyVector<mValueType, DIM>;

  /** constructor
   *@param spos the single-particle orbital set
   *@param first index of the first particle
   */
  DiracDeterminantBatched(SPOSetPtr const spos, int first = 0);

  // copy constructor and assign operator disabled
  DiracDeterminantBatched(const DiracDeterminantBatched& s) = delete;
  DiracDeterminantBatched& operator=(const DiracDeterminantBatched& s) = delete;

  /** set the index of the first particle in the determinant and reset the size of the determinant
   *@param first index of first particle
   *@param nel number of particles in the determinant
   */
  void set(int first, int nel, int delay = 1) override final;

  void evaluateDerivatives(ParticleSet& P,
                           const opt_variables_type& active,
                           std::vector<ValueType>& dlogpsi,
                           std::vector<ValueType>& dhpsioverpsi) override;

  ///reset the size: with the number of particles and number of orbtials
  void resize(int nel, int morb);

  void registerData(ParticleSet& P, WFBufferType& buf) override;

  void updateAfterSweep(ParticleSet& P, ParticleSet::ParticleGradient_t& G, ParticleSet::ParticleLaplacian_t& L);

  LogValueType updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch = false) override;

  void copyFromBuffer(ParticleSet& P, WFBufferType& buf) override;

  /** return the ratio only for the  iat-th partcle move
   * @param P current configuration
   * @param iat the particle thas is being moved
   */
  PsiValueType ratio(ParticleSet& P, int iat) override;

  //Ye: TODO, good performance needs batched SPO evaluation.
  //void mw_calcRatio(const std::vector<WaveFunctionComponent*>& WFC_list,
  //                  const std::vector<ParticleSet*>& P_list,
  //                  int iat,
  //                  std::vector<PsiValueType>& ratios) override;

  /** compute multiple ratios for a particle move
   */
  void evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios) override;

  void mw_evaluateRatios(const RefVector<WaveFunctionComponent>& wfc_list,
                         const RefVector<const VirtualParticleSet>& vp_list,
                         std::vector<std::vector<ValueType>>& ratios) override;

  PsiValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) override;

  void mw_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                    const std::vector<ParticleSet*>& P_list,
                    int iat,
                    std::vector<PsiValueType>& ratios,
                    std::vector<GradType>& grad_new) override;

  GradType evalGrad(ParticleSet& P, int iat) override;

  GradType evalGradSource(ParticleSet& P, ParticleSet& source, int iat) override;

  GradType evalGradSource(ParticleSet& P,
                          ParticleSet& source,
                          int iat,
                          TinyVector<ParticleSet::ParticleGradient_t, OHMMS_DIM>& grad_grad,
                          TinyVector<ParticleSet::ParticleLaplacian_t, OHMMS_DIM>& lapl_grad) override;

  /** move was accepted, update the real container
   */
  void acceptMove(ParticleSet& P, int iat, bool safe_to_delay = false) override;

  void mw_acceptMove(const std::vector<WaveFunctionComponent*>& WFC_list,
                     const std::vector<ParticleSet*>& P_list,
                     int iat, bool safe_to_delay = false) override
  {
    for (int iw = 0; iw < WFC_list.size(); iw++)
      WFC_list[iw]->acceptMove(*P_list[iw], iat, safe_to_delay);
  }

  void completeUpdates() override;

  void mw_completeUpdates(const std::vector<WaveFunctionComponent*>& WFC_list) override
  {
    for (int iw = 0; iw < WFC_list.size(); iw++)
      WFC_list[iw]->completeUpdates();
  }

  /** move was rejected. copy the real container to the temporary to move on
   */
  void restore(int iat) override;

  ///evaluate log of a determinant for a particle set
  LogValueType evaluateLog(ParticleSet& P,
                           ParticleSet::ParticleGradient_t& G,
                           ParticleSet::ParticleLaplacian_t& L) override;

  //Ye: TODO, good performance needs batched SPO evaluation.
  //void mw_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
  //                    const std::vector<ParticleSet*>& P_list,
  //                    const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
  //                    const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list) override;

  void recompute(ParticleSet& P) override;

  void evaluateHessian(ParticleSet& P, HessVector_t& grad_grad_psi) override;

  /** cloning function
   * @param tqp target particleset
   * @param spo spo set
   *
   * This interface is exposed only to SlaterDet and its derived classes
   * can overwrite to clone itself correctly.
   */
  DiracDeterminantBatched* makeCopy(SPOSet* spo) const override;

  void evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios) override;

  /// psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
  ValueMatrix_t psiM_temp;

  /// inverse transpose of psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
  ValueMatrix_t psiMinv;

  /// dpsiM(i,j) \f$= \nabla_i \psi_j({\bf r}_i)\f$
  GradMatrix_t dpsiM;

  /// d2psiM(i,j) \f$= \nabla_i^2 \psi_j({\bf r}_i)\f$
  ValueMatrix_t d2psiM;

  /// Used for force computations
  GradMatrix_t grad_source_psiM, grad_lapl_source_psiM;
  HessMatrix_t grad_grad_source_psiM;

  GradMatrix_t phi_alpha_Minv, grad_phi_Minv;
  ValueMatrix_t lapl_phi_Minv;
  HessMatrix_t grad_phi_alpha_Minv;

  /// value of single-particle orbital for particle-by-particle update
  ValueVector_t psiV;
  GradVector_t dpsiV;
  ValueVector_t d2psiV;

  /// delayed update engine
  DET_ENGINE_TYPE det_engine_;

  PsiValueType curRatio;

private:
  /// invert psiM or its copies
  void invertPsiM(const ValueMatrix_t& logdetT, ValueMatrix_t& invMat);

  /// Resize all temporary arrays required for force computation.
  void resizeScratchObjectsForIonDerivs();

  /// internal function computing ratio and gradients after computing the SPOs, used by ratioGrad.
  PsiValueType ratioGrad_compute(int iat, GradType& grad_iat);

};

extern template class DiracDeterminantBatched<>;
#if defined(ENABLE_CUDA)
//extern template class DiracDeterminantBatched<DelayedUpdateCUDA<QMCTraits::ValueType, QMCTraits::QTFull::ValueType>>;
#endif

} // namespace qmcplusplus
#endif
