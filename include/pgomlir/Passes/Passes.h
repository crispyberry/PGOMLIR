#ifndef PGOMLIR_PASSES_H
#define PGOMLIR_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
class RewritePatternSet;
namespace pgomlir {
std::unique_ptr<Pass> createSettledAttrToSCFPass();
std::unique_ptr<Pass> createBranchProbabilityInfoPass();
void populateSCFToCFConversionPatterns(RewritePatternSet &patterns);
std::unique_ptr<Pass> createSCFToCFPass();
} // namespace pgomlir
} // namespace mlir

#endif // PGOMLIR_PASSES_H