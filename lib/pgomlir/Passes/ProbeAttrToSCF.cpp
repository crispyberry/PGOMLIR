#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pgomlir/Passes/Passes.h"
#include "pgomlir/Utilities/GetConstantVerify.h"

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

// struct SelectExprAttrArithPattern : public OpRewritePattern<arith::SelectOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(arith::SelectOp selectOp,
//                                 PatternRewriter &rewriter) const override;
// };

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
  std::string lower = getConstantVerify(forOp.getLowerBound());
  std::string step = getConstantVerify(forOp.getStep());

  std::string trip = upper + "," + lower + "," + step;
  auto tripCountAttr = StringAttr::get(forOp.getContext(), trip);

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
  if (ifOp->getAttrOfType<StringAttr>("ifExpr"))
    return failure();
  // Deal with CmpFOp ...
  auto cmpIOp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  auto predicate = arith::stringifyEnum(cmpIOp.getPredicate()).str();
  std::string lhsvalue = getConstantVerify(cmpIOp.getLhs());
  std::string rhsvalue = getConstantVerify(cmpIOp.getRhs());
  // TODO: Update this code by designing a util::getDefiningWithContol to
  // identify the data with control flow.

  std::string expr = lhsvalue + " "+predicate + " "+rhsvalue;
  auto ifExprAttr = StringAttr::get(ifOp.getContext(), expr);
  rewriter.updateRootInPlace(
      ifOp, [&]() { ifOp->setAttr("ifExpr", ifExprAttr); });
  return success();
}

// LogicalResult
// SelectExprAttrArithPattern::matchAndRewrite(arith::SelectOp selectOp,
//                                             PatternRewriter &rewriter) const {
//   if (selectOp->getAttrOfType<StringAttr>("selectExpr"))
//     return failure();
//   auto condition = selectOp.getCondition();
//   auto cmpIOp = condition.getDefiningOp<arith::CmpIOp>();
//   auto predicate = arith::stringifyEnum(cmpIOp.getPredicate()).str();
//   std::string lhsvalue = getConstantVerify(cmpIOp.getLhs());
//   std::string rhsvalue = getConstantVerify(cmpIOp.getRhs());

//   auto trueOperandValue = selectOp.getTrueValue();
//   auto falseOperandValue = selectOp.getFalseValue();
//   std::string truevalue = getConstantVerify(trueOperandValue);
//   std::string falsevalue = getConstantVerify(falseOperandValue);

//   std::string expr = predicate + "," + lhsvalue + "," + rhsvalue + ":" +
//                      truevalue +","+ falsevalue;
//   auto selectExprAttr = StringAttr::get(selectOp.getContext(), expr);
//   rewriter.updateRootInPlace(
//       selectOp, [&]() { selectOp->setAttr("selectExpr", selectExprAttr); });
//   return success();
// }

void ProbeAttrToSCFPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<TripCountAttrSCFPattern>(&getContext());
  patterns.add<ComparisonExprAttrSCFPattern>(&getContext());
  //patterns.add<SelectExprAttrArithPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
std::unique_ptr<Pass> mlir::pgomlir::createProbeAttrToSCFPass() {
  return std::make_unique<ProbeAttrToSCFPass>();
}