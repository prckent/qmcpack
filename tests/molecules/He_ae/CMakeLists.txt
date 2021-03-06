# Deterministic test for QMC optimizer 

IF (MIXED_PRECISION OR QMC_COMPLEX)
  MESSAGE_VERBOSE("Skipping optimizer deterministic He tests in mixed precision (QMC_MIXED_PRECISION=1) or complex (QMC_COMPLEX=1)")
ELSEIF (QMC_CUDA)
  MESSAGE_VERBOSE("Skipping optimizer deterministic He tests because CUDA is not supported (QMC_CUDA=1)")
ELSE()
  SIMPLE_RUN_AND_CHECK(
    deterministic-He_ae-opt
    "${CMAKE_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt.xml
    1 1
    check_opt.py -s 1 -p det_He_opt -r qmc-ref/det_He_opt.s001.opt.xml
    )

 LIST(APPEND He_VMC_SCALARS "totenergy" "-3.25291030 0.000001")

   QMC_RUN_AND_CHECK(deterministic-He_ae-opt_vmc
                    "${CMAKE_SOURCE_DIR}/tests/molecules/He_ae"
                    det_He_opt
                    det_He_opt.xml
                    1 1
                    TRUE
                    2 He_VMC_SCALARS # VMC
                    )
ENDIF()

