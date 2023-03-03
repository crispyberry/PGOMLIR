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
// struct MyAttribute {
//   static StringAttr get(MLIRContext *context, llvm::StringRef st) {
//     return StringAttr::get(context, st);
//   }
// };

struct TripCountAttrSCFPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

struct ComparisonExprAttrSCFPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
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
      // TODO:This demo part is for dynamic step value, mayebe later we don't
      // have to add this information as attribute. Just use this information
      // for analysis. Or just add dynamic information as unknown, and deal with
      // unknown by other pass?
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
// TODO: We should think about what kind of ops and operands may work with
// control flow! And maybe we can build a util to identify the things that are
// control flow sensitive! At first, if the operation is not a constant op or
// cannot getDefiningOp, it can be. If the type of operands is index, it likely
// can be.
LogicalResult
ComparisonExprAttrSCFPattern::matchAndRewrite(scf::IfOp ifOp,
                                              PatternRewriter &rewriter) const {
  if (ifOp->getAttrOfType<StringAttr>("comparisonExpr"))
    return failure();
  auto cmpiOp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  auto predicate = arith::stringifyEnum(cmpiOp.getPredicate()).str();
  std::string lhsvalue;
  std::string rhsvalue;
  // TODO: Update this code by designing a util::getDefiningWithContol to
  // identify the data with control flow.
  if (auto rhs = cmpiOp.getRhs().getDefiningOp<arith::ConstantIntOp>()) {
    rhsvalue = llvm::utostr(rhs.value());
    if (auto definelhs = cmpiOp.getLhs().getDefiningOp<arith::IndexCastOp>()) {
      if (auto producer = definelhs.getIn().getDefiningOp()) {
        lhsvalue = producer->getName().getStringRef().str();
      } else {
        auto blockArg = definelhs.getIn().cast<BlockArgument>();
        lhsvalue = "blockArgIndex:" + llvm::utostr(blockArg.getArgNumber());
      }
    }
  }

  std::string expr = predicate + " " + lhsvalue + " " + rhsvalue;
  auto comparisonExprAttr = StringAttr::get(ifOp.getContext(), expr);
  rewriter.updateRootInPlace(
      ifOp, [&]() { ifOp->setAttr("comparisonExpr", comparisonExprAttr); });
  return failure();
}

void ProbeAttrToSCFPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<TripCountAttrSCFPattern>(&getContext());
  patterns.add<ComparisonExprAttrSCFPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
std::unique_ptr<Pass> mlir::pgomlir::createProbeAttrToSCFPass() {
  return std::make_unique<ProbeAttrToSCFPass>();
}