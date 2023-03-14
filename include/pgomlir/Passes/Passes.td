#ifndef PGO_MLIR_PASSES_H
#define PGO_MLIR_PASSES_H

include "mlir/Pass/PassBase.td"

def SettledAttrToSCFPass : Pass<"settled-attr-to-scf-pass"> {
  let summary = "Add settled information atrributes to SCF operations";
  let constructor = "mlir::pgomlir::createSettledAttrToSCFPass()";
}

def BranchProbabilityInfoPass : Pass<"branch-prob-info-pass">{
  let summary = "Reasoning branch probability information by settled information";
  let constructor = "mlir::pgomlir::createBranchProbabilityInfoPass()";
}

def SCFToControlFlow : Pass<"convert-scf-to-cf-with-info"> {
  let summary = "Convert SCF dialect to ControlFlow dialect with extra information, replacing structured"
                " control flow with a CFG";
  let constructor = "mlir::createSCFToCFPass()";
}
#endif // PGO_MLIR_PASSES_H