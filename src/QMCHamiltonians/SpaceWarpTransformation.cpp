#include "QMCHamiltonians/SpaceWarpTransformation.h"
#include "Particle/DistanceTableData.h"
#include "type_traits/scalar_traits.h"
namespace qmcplusplus
{

SpaceWarpTransformation::SpaceWarpTransformation(ParticleSet& elns, ParticleSet& ions):
      myTableIndex(elns.addTable(ions))
{
  Nelec=elns.getTotalNum();
  Nions=ions.getTotalNum(); 
  warpval.resize(Nelec,Nions);
  gradval.resize(Nelec,Nions);
  app_log()<<"SpaceWarp transformation with 1/r^4 initialized\n"; 
}

SpaceWarpTransformation::RealType SpaceWarpTransformation::f(RealType r)
{
  return 1.0/(r*r*r*r);
}

SpaceWarpTransformation::RealType SpaceWarpTransformation::df(RealType r)
{
  return -4.0/(r*r*r*r*r);
}

void SpaceWarpTransformation::computeSWTIntermediates(ParticleSet& P, ParticleSet& ions)
{
  const DistanceTableData& d_ab(P.getDistTable(myTableIndex));
  for (size_t iel = 0; iel < Nelec; ++iel)
  {
    const auto& dist = d_ab.getDistRow(iel);
    const auto& dr   = d_ab.getDisplRow(iel);
    for (size_t ionid = 0; ionid < Nions; ++ionid)
    {
      warpval[iel][ionid]=f(dist[ionid]);
      gradval[iel][ionid]=-dr[ionid]*(df(dist[ionid])/dist[ionid]); //because there's a -1 in distance table displacement definition.  R-r :(. 
    }
  }

}

void SpaceWarpTransformation::getSWT(int iat, ParticleScalar_t& w, Force_t& grad_w)
{
  for(size_t iel=0; iel<Nelec; iel++)
  {
    RealType warpdenom=0.0;
    PosType  denomgrad=0.0;
    for(size_t ionid=0; ionid<Nions; ionid++)
    {
      warpdenom+=warpval[iel][ionid];
      denomgrad+=gradval[iel][ionid];
    }
    w[iel]=warpval[iel][iat]/warpdenom;
    grad_w[iel]=gradval[iel][iat]/warpdenom - w[iel]*denomgrad/warpdenom;
  }
}

void SpaceWarpTransformation::computeSWT(ParticleSet& P, ParticleSet& ions, Force_t& dEl, ParticleGradient_t& dlogpsi, Force_t& el_contribution, Force_t& psi_contribution)
{
  el_contribution=0;
  psi_contribution=0;
  ParticleScalar_t w;
  Force_t gradw;
  w.resize(Nelec);
  gradw.resize(Nelec);

  PosType gwfn=0;
  computeSWTIntermediates(P,ions);
    
  for(size_t iat=0; iat<Nions; iat++)
  {
    getSWT(iat,w,gradw);
    
    for(size_t iel=0; iel<Nelec; iel++)
    {
      el_contribution+=w[iel]*dEl[iel];

      #if defined(QMC_COMPLEX)
      convert(dlogpsi[iel],gwfn);
      #else
      gwfn=dlogpsi[iel];
      #endif 
      psi_contribution[iat]=w[iel]*gwfn+0.5*gradw[iel];
    }
    
  }
}

/*void SpaceWarpTransformation::sw(ParticleSet& P, ParticleSet& ions, int iat, ParticleScalar_t& sw_vals, Force_t& sw_grads)
{
  int Nelec=P.getTotalNum();
  int Nions=ions.getTotalNum();
  const DistanceTableData& d_ab(P.getDistTable(myTableIndex));
  for (size_t iel = 0; iel < Nelec; ++iel)
  {
    const auto& dist = d_ab.getDistRow(iel);
    const auto& dr   = d_ab.getDisplRow(iel);
    mRealType esum   = czero;
    for (size_t ionid = 0; ionid < ionid; ++ionid)
    {
     
    }
  }
}*/

}
