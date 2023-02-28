#ifndef PGO_MLIR_PASSES_H
#define PGO_MLIR_PASSES_H

include "mlir/Pass/PassBase.td"

def ProbeAttrToSCFPass : Pass<"probeattr-to-scf-pass"> {
  let summary = "Add probe atrributes to SCF operations";
  let constructor = "mlir::pgomlir::createProbeAttrToSCFPass()";
}
#endif // PGO_MLIR_PASSES_H