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
#endif // PGO_MLIR_PASSES_H