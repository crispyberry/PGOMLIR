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
struct MyAttribute {
  static StringAttr get(MLIRContext *context, llvm::StringRef st) {
    return StringAttr::get(context, st);
  }
};

struct TripCountAttrSCFPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

struct ProbeAttrToSCFPass
    : public PassWrapper<ProbeAttrToSCFPass, OperationPass<>> {
  void runOnOperation() override;
};
} // namespace

LogicalResult
TripCountAttrSCFPattern::matchAndRewrite(scf::ForOp forOp,
                                         PatternRewriter &rewriter) const {
  if (forOp->getAttrOfType<StringAttr>("tripCount"))
    return failure();

  // Firstly assume that bounds for loop are defined well by operations within
  // arith.
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

  if (!lbCstOp && !ubCstOp && !stepCstOp) {
    auto tripCountAttr =
        StringAttr::get(forOp.getContext(), std::string("unknown"));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (lbCstOp && ubCstOp && !stepCstOp) {
    std::string stepName;
    if (Operation *producer = forOp.getStep().getDefiningOp()) {
      stepName = "op:" + producer->getName().getStringRef().str();
    } else {
      // If there is no defining op, the Value is necessarily a Block
      // argument.
      auto blockArg = forOp.getStep().cast<BlockArgument>();
      stepName = "blockArgIndex:" + llvm::utostr(blockArg.getArgNumber());
    }

    auto tripCountAttr = StringAttr::get(
        forOp.getContext(), llvm::utostr(ubCstOp.value()) + std::string("-") +
                                llvm::utostr(lbCstOp.value()) +
                                std::string("/") + stepName);
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (!lbCstOp && ubCstOp && stepCstOp) {
    auto tripCountAttr = StringAttr::get(
        forOp.getContext(), llvm::utostr(ubCstOp.value()) + std::string("-") +
                                std::string("unknown") + std::string("/") +
                                llvm::utostr(stepCstOp.value()));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (lbCstOp && !ubCstOp && stepCstOp) {
    auto tripCountAttr =
        StringAttr::get(forOp.getContext(),
                        std::string("unknown") + std::string("-") +
                            llvm::utostr(lbCstOp.value()) + std::string("/") +
                            llvm::utostr(stepCstOp.value()));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (!lbCstOp && !ubCstOp && stepCstOp) {
    auto tripCountAttr = StringAttr::get(
        forOp.getContext(), std::string("unknown") + std::string("/") +
                                llvm::utostr(stepCstOp.value()));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (lbCstOp && !ubCstOp && !stepCstOp) {
    auto tripCountAttr = StringAttr::get(
        forOp.getContext(), std::string("unknown") + std::string("-") +
                                llvm::utostr(lbCstOp.value()) +
                                std::string("/") + std::string("unknown"));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (!lbCstOp && ubCstOp && !stepCstOp) {
    auto tripCountAttr = StringAttr::get(
        forOp.getContext(), llvm::utostr(ubCstOp.value()) + std::string("-") +
                                std::string("unknown") + std::string("/") +
                                std::string("unknown"));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  } else if (lbCstOp.value() < 0 || ubCstOp.value() < 0 ||
             stepCstOp.value() < 0) {
    auto tripCountAttr =
        StringAttr::get(forOp.getContext(),
                        llvm::utostr(ubCstOp.value()) + std::string("-") +
                            llvm::utostr(lbCstOp.value()) + std::string("/") +
                            llvm::utostr(stepCstOp.value()));
    rewriter.updateRootInPlace(
        forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });
    return failure();
  }

  int64_t tripCount =
      ceilDiv(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
  auto tripCountAttr =
      StringAttr::get(forOp.getContext(), llvm::utostr(tripCount));

  rewriter.updateRootInPlace(
      forOp, [&]() { forOp->setAttr("tripCount", tripCountAttr); });

  return success();
}

void ProbeAttrToSCFPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<TripCountAttrSCFPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
std::unique_ptr<Pass> mlir::pgomlir::createProbeAttrToSCFPass() {
  return std::make_unique<ProbeAttrToSCFPass>();
}