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

def SCFToCFPass : Pass<"convert-scf-to-cf-with-info"> {
  let summary = "Convert SCF dialect to ControlFlow dialect with extra information, replacing structured"
                " control flow with a CFG";
  let constructor = "mlir::pgomlir::createSCFToCFPass()";
}

// def SettledAttrToMetadataPass : Pass<"convert-settled-information-to-llvm-metadata"> {
//   let summary = "Convert settled information as attributes of scf operations to llvm.metadata"
//   let constructor = "mlir::pgomlir::createSettledAttrToMetadataPass()";
// }

def CFToLLVMWithAttr : Pass<"convert-cf-to-llvm-with-attr", "mlir::ModuleOp"> {
  let summary = "Convert ControlFlow operations to the LLVM dialect with attribute";
  let description = [{
    Convert ControlFlow operations into LLVM IR dialect operations.

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.
  }];
  let constructor = "mlir::pgomlir::createCFToLLVMWithAttrPass()";
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

#endif // PGO_MLIR_PASSES_H