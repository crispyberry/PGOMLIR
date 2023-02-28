add_mlir_dialect_library(MLIRPGOTransforms
  ProbeAttrToSCF.cpp

  ${PROJECT_SOURCE_DIR}/include/pgomlir/Passes

  DEPENDS
  MLIRPGOPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRArithDialect
  MLIRSupport
)