#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace pgomlir;

namespace {
struct AddMetadataToBlocksPass
    : public PassWrapper<AddMetadataToBlocksPass, OperationPass<>> {
  void runOnOperation() override {
    auto ops = getOperation();
    bool inLoopRegion = false;
    ops->walk([&](Operation *op) {
      if (auto funcOp = dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(op)) {
        for (auto &block : funcOp) {
          bool skipBlock = false;
          for (auto &op : block) {
            if (auto brOp = dyn_cast<mlir::LLVM::BrOp>(op)) {
              // Check for llvm.br with loop attribute
              if (brOp->hasAttr("loop")) {
                inLoopRegion = !inLoopRegion;
                skipBlock = true;
              }
            }
          }
          if (skipBlock) {
            continue;
          }
          // If we are inside the loop region, add metadata to the block using
          // llvm.metadata operation
          if (inLoopRegion) {
            OpBuilder builder(&block, block.begin());
            llvm::errs() << block.front() << "\n";
            // auto firstOpLoc = block.getTerminator();
            llvm::errs() << "Find Place!\n";
            auto metadata = builder.create<mlir::LLVM::MetadataOp>(
                builder.getUnknownLoc(), builder.getStringAttr("pgo-metadata"));
          }
        }
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::pgomlir::createAddMetadataToBlocksPass() {
  return std::make_unique<AddMetadataToBlocksPass>();
}
