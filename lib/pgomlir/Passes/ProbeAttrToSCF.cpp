#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pgomlir/Passes/Passes.h"
#include "pgomlir/Utilities/GetConstant.h"

using namespace mlir;
using namespace pgomlir;

namespace {

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
  std::string upper = getConstantVerify(forOp.getUpperBound());
  std::string  lower = getConstantVerify(forOp.getLowerBound());
  std::string  step = getConstantVerify(forOp.getStep());

  std::string trip = upper+","+lower+","+step;
  auto tripCountAttr =
      StringAttr::get(forOp.getContext(), trip);

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
  std::string lhsvalue = getConstantVerify(cmpiOp.getLhs());
  std::string rhsvalue = getConstantVerify(cmpiOp.getRhs());
  // TODO: Update this code by designing a util::getDefiningWithContol to
  // identify the data with control flow.

  std::string expr = predicate + "," + lhsvalue + "," + rhsvalue;
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