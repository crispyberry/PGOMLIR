add_mlir_dialect_library(MLIRPGOTransforms
  SettledAttrToSCF.cpp
  BranchProbabilityInfo.cpp
  SCFToCF.cpp

  ${PROJECT_SOURCE_DIR}/include/pgomlir/Passes

  DEPENDS
  MLIRPGOPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTransforms
)