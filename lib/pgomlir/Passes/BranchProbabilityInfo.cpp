#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pgomlir/Passes/Passes.h"


using namespace mlir;
using namespace pgomlir;
namespace {
struct BranchProbabilityInfoPass
    : public PassWrapper<BranchProbabilityInfoPass, OperationPass<>> {
  void runOnOperation() override;
};

} // namespace

void BranchProbabilityInfoPass::runOnOperation() {
  auto ops = getOperation();
  auto &context = getContext();
  ops->walk([&](Operation* op) {
    if(auto ifOp = llvm::dyn_cast_or_null<scf::IfOp>(op)){
      if(auto ifExprAttr = ifOp->getAttrOfType<StringAttr>("ifExpr")){
        llvm::errs()<<"Found attribute ifExpr with value " << ifExprAttr.getValue() << "\n";
      }
    }
  });
}

std::unique_ptr<Pass> mlir::pgomlir::createBranchProbabilityInfoPass() {
  return std::make_unique<BranchProbabilityInfoPass>();
}