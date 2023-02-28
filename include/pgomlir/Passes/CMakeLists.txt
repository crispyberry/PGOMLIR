set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PGOMLIR)
add_public_tablegen_target(MLIRPGOPassIncGen)

add_mlir_doc(Passes MLIRPGOPasses ./ -gen-pass-doc)