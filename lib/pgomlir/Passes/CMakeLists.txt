add_mlir_dialect_library(MLIRPGOTransforms
  SettledAttrToSCF.cpp
  BranchProbabilityInfo.cpp
  SCFToCF.cpp
  CFToLLVMWithAttr.cpp
  AddMetadataToBlocks.cpp

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
  MLIRAnalysis
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransformUtils
)