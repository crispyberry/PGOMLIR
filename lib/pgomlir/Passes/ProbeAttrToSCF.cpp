#include "PassDetails.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/MathExtras.h"

#include "pgomlir/Passes/Passes.h"

using namespace mlir;
using namespace pgomlir;

namespace {
struct MyAttribute {
  static StringAttr get(MLIRContext *context, llvm::StringRef st) {
    return StringAttr::get(context, st);
  }
};

struct ProbeAttrToSCFPass
    : public PassWrapper<ProbeAttrToSCFPass, OperationPass<>> {
  void runOnOperation() override;
};
} // namespace

void ProbeAttrToSCFPass::runOnOperation() {
  auto ops = getOperation();
  auto &context = getContext();
  ops->walk([&](Operation* op) {
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp)
      return;

    auto lbCstOp =
        forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto ubCstOp =
        forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!lbCstOp || !ubCstOp || !stepCstOp || lbCstOp.value() < 0 ||
        ubCstOp.value() < 0 || stepCstOp.value() < 0)
      return;
    int64_t tripCount =
        mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
    //Create the custom attribute.
    StringAttr attr = MyAttribute::get(&context, llvm::utostr(tripCount));

    // Set the custom attribute on the `scf.for` operation.
    (*forOp).setAttr("my_attribute", attr);
  });
}

std::unique_ptr<Pass> mlir::pgomlir::createProbeAttrToSCFPass() {
  return std::make_unique<ProbeAttrToSCFPass>();
}