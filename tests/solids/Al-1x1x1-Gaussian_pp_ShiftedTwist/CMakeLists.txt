
IF(ENABLE_SOA)
 IF (NOT QMC_CUDA)
   IF (QMC_COMPLEX)

# VMC long run with no Jastrows 
# Compare directly to Hartree Fock Energy of: E=-8.072494830814307
#Data generated using utils/make_ref_data.sh  Al-1x1x1-Gaussian_pp_Tw_cplx_LONG qmc_ref/qmc_long.s000.scalar.dat  qmc_ref/qmc_ref_long.s000.scalar.dat 
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "totenergy" "-8.072494830814307 0.014895296777")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "variance" "0.929704520713 0.035362222511")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "kinetic" "3.564934899261 0.010603465208")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "potential" "-11.637571841025 0.015954075009")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "eeenergy" "-3.109468568661 0.014478131707")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "localecp" "0.693022086548 0.015917294909")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "nonlocalecp" "1.591560552665 0.016677475370")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS "ionion" "-10.812685912000 0.020000000000")
  QMC_RUN_AND_CHECK(long-Al-1x1x1-Gaussian-pp-cplx
                    "${CMAKE_SOURCE_DIR}/tests/solids/Al-1x1x1-Gaussian_pp_ShiftedTwist"
                    vmc_long 
                    vmc_long.in.xml
                    1 16
                    TRUE
                    0 Al-1x1x1-Gaussian_pp_Tw_cplx_LONG_SCALARS #VMC 
                    )

# VMC short run with no Jastrows 
# Compare directly to Hartree Fock Energy of: E=-8.072494830814307
#Data generated using utils/make_ref_data.sh  Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT qmc_ref/qmc_short.s000.scalar.dat  qmc_ref/qmc_ref_long.s000.scalar.dat 
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "totenergy" "-8.072494830814307 0.025268003771")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "variance" "0.929704520713 0.046252853972")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "kinetic" "3.564934899261 0.015825393834")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "potential" "-11.637571841025 0.029573348064")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "eeenergy" "-3.109468568661 0.026007614629")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "localecp" "0.693022086548 0.049611353506")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "nonlocalecp" "1.591560552665 0.047099343376")
LIST(APPEND Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS "ionion" "-10.812685912000 0.020000000000")
  QMC_RUN_AND_CHECK(short-Al-1x1x1-Gaussian-pp-cplx
                    "${CMAKE_SOURCE_DIR}/tests/solids/Al-1x1x1-Gaussian_pp_ShiftedTwist"
                    vmc_short 
                    vmc_short.in.xml
                    1 16
                    TRUE
                    0 Al-1x1x1-Gaussian_pp_Tw_cplx_SHORT_SCALARS #VMC 
                    )


    ELSE()
      MESSAGE_VERBOSE("Skipping Complex Periodic LCAO  if Complex code not build (QMC_COMPLEX=0)")
    ENDIF()
  ELSE()
    MESSAGE_VERBOSE("Skipping Periodic LCAO as not supported by CUDA build (QMC_CUDA=1)")
  ENDIF()
ELSE()
  MESSAGE_VERBOSE("Skipping Periodic LCAO as not supported by AoS build (ENABLE_SOA=0)")
ENDIF()
