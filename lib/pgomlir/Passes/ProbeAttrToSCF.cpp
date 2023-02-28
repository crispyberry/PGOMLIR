#include "PassDetails.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "pgomlir/Passes/Passes.h"

using namespace mlir;
using namespace pgomlir;

namespace {
struct MyAttribute {
  static StringAttr get(MLIRContext *context) {
    return StringAttr::get(context, "my_string_value");
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
  ops->walk([&](scf::ForOp forOp) {
    // Create the custom attribute.
    StringAttr attr = MyAttribute::get(&context);

    // Set the custom attribute on the `scf.for` operation.
    forOp->setAttr("my_attribute", attr);
  });
}

std::unique_ptr<Pass> mlir::pgomlir::createProbeAttrToSCFPass() {
  return std::make_unique<ProbeAttrToSCFPass>();
}