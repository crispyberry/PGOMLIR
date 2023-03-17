#ifndef PGOMLIR_PASSES_H
#define PGOMLIR_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
class RewritePatternSet;
class LLVMTypeConverter;
class ModuleOp;
namespace pgomlir {
std::unique_ptr<Pass> createSettledAttrToSCFPass();
std::unique_ptr<Pass> createBranchProbabilityInfoPass();
void populateSCFToCFConversionPatterns(RewritePatternSet &patterns);
std::unique_ptr<Pass> createSCFToCFPass();
//std::unique_ptr<Pass> createSettledAttrToMetadataPass();
void populateControlFlowToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns);

/// Creates a pass to convert the ControlFlow dialect into the LLVMIR dialect.
std::unique_ptr<Pass> createCFToLLVMWithAttrPass();
} // namespace pgomlir
} // namespace mlir

#endif // PGOMLIR_PASSES_H