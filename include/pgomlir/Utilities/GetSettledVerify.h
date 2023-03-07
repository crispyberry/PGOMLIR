#ifndef GET_SETTLED_H
#define GET_SETTLED_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

std::string getYieldVerify(Value value) {
  llvm::errs() << "yiled: " << value << "\n";
  while (auto producer = value.getDefiningOp()) {
    llvm::errs() << producer->getName() << "\n";
    if (auto cosntantIndexOp =
            llvm::dyn_cast_or_null<arith::ConstantIndexOp>(producer)) {
      return std::to_string(cosntantIndexOp.value());
    } else if (auto cosntantIntOp =
                   llvm::dyn_cast_or_null<arith::ConstantIntOp>(producer)) {
      return std::to_string(cosntantIntOp.value());
    } else if (auto constantIndexCastOp =
                   llvm::dyn_cast_or_null<arith::IndexCastOp>(producer)) {
      value = constantIndexCastOp.getIn();
      // Further we should deal with other binary op like AddFOP DivOP ...
    } else if (auto constantAddIOp =
                   llvm::dyn_cast_or_null<arith::AddIOp>(producer)) {
      auto left = constantAddIOp.getLhs();
      auto right = constantAddIOp.getRhs();
      return getYieldVerify(left) + "+" + getYieldVerify(right);
    } else if (auto selectOp =
                   llvm::dyn_cast_or_null<arith::SelectOp>(producer)) {
      auto condition = selectOp.getCondition();
      auto cmpIOp = condition.getDefiningOp<arith::CmpIOp>();
      auto predicate = arith::stringifyEnum(cmpIOp.getPredicate()).str();
      std::string lhsvalue = getYieldVerify(cmpIOp.getLhs());
      std::string rhsvalue = getYieldVerify(cmpIOp.getRhs());

      auto trueOperandValue = selectOp.getTrueValue();
      auto falseOperandValue = selectOp.getFalseValue();
      std::string truevalue = getYieldVerify(trueOperandValue);
      std::string falsevalue = getYieldVerify(falseOperandValue);
      std::string expr;
      for (Operation *userOp : condition.getUsers()) {
        if (userOp->getName().getStringRef().str() == "scf.if") {
          expr = std::string("{ if") + " ? " + truevalue + " : " + falsevalue +
                 "}";
          break;
        } else {
          expr = "{ " + lhsvalue + " " + predicate + " " + rhsvalue + ":" +
                 truevalue + "," + falsevalue + "}";
        }
      }
      return expr;
    } else if (auto ifOp = llvm::dyn_cast_or_null<scf::IfOp>(producer)) {
      auto yieldOpInIfThen = dyn_cast_or_null<scf::YieldOp>(
          ifOp.getThenRegion().back().getTerminator());
      auto yieledValue1 = yieldOpInIfThen.getResults()[0];
      std::string thenval = " if ? " + getYieldVerify(yieledValue1);
      auto yieldOpInIfElse = dyn_cast_or_null<scf::YieldOp>(
          ifOp.getElseRegion().back().getTerminator());
      auto yieledValue2 = yieldOpInIfElse.getResults()[0];
      std::string elseval = getYieldVerify(yieledValue2);
      return "{" + thenval + ":" + elseval + "} ";

    } else {
      return std::string("unknow");
    }
  }
  if (auto blockArg = value.cast<BlockArgument>()) {
    std::string parentOp;
    std::string iterExpr;
    std::string yiledExpr;

    auto posInArg = blockArg.getArgNumber();
    if (isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      parentOp = std::string("Func");
      iterExpr = "";
    } else if (isa<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      parentOp = std::string("For");

      auto forOp = llvm::dyn_cast_or_null<scf::ForOp>(
          blockArg.getOwner()->getParentOp());
      if (posInArg != 0) { // if it is in iter_args.
        auto iterInitValue = forOp.getInitArgs()[posInArg - 1];
        iterExpr = "(" + getYieldVerify(iterInitValue) + ")";
        // scf.for binds iter_args with scf.yield, so we must take scf.yield
        // into consideration.
      }
    }
    return std::string("Index" + parentOp + std::to_string(posInArg) +
                       iterExpr + yiledExpr);
  }
  return std::string("unknow");
}

std::string getSettledVerify(Value value) {
  llvm::errs() << "not yiled: " << value << "\n";
  while (auto producer = value.getDefiningOp()) {
    if (auto cosntantIndexOp =
            llvm::dyn_cast_or_null<arith::ConstantIndexOp>(producer)) {
      return std::to_string(cosntantIndexOp.value());
    } else if (auto cosntantIntOp =
                   llvm::dyn_cast_or_null<arith::ConstantIntOp>(producer)) {
      return std::to_string(cosntantIntOp.value());
    } else if (auto constantIndexCastOp =
                   llvm::dyn_cast_or_null<arith::IndexCastOp>(producer)) {
      value = constantIndexCastOp.getIn();
    } else if (auto constantAddIOp =
                   llvm::dyn_cast_or_null<arith::AddIOp>(producer)) {
      auto left = constantAddIOp.getLhs();
      auto right = constantAddIOp.getRhs();
      return getSettledVerify(left) + "+" + getSettledVerify(right);
    } else { // TODO: Further we should deal with other binary op like AddFOP
             // DivOP ...
      return std::string("unknow");
    }
  }
  if (auto blockArg = value.cast<BlockArgument>()) {
    std::string parentOp = "";
    std::string ivExpr = "";
    std::string iterExpr = "";
    std::string yiledExpr = "";

    auto posInArg = blockArg.getArgNumber();
    if (isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      parentOp = std::string("Func");
    } else if (isa<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      parentOp = std::string("For");

      auto forOp = llvm::dyn_cast_or_null<scf::ForOp>(
          blockArg.getOwner()->getParentOp());
        llvm::errs()<<"POS:"<<posInArg<<"\n";
      if (posInArg != 0) { // If it is in iter_args.
        auto iterInitValue = forOp.getInitArgs()[posInArg - 1];
        iterExpr = "(" + getSettledVerify(iterInitValue) + ")";
        // scf.for binds iter_args with scf.yield, so we must take scf.yield
        // into consideration.
        if (auto yieldOpInFor = dyn_cast_or_null<scf::YieldOp>(
                forOp.getLoopBody().back().getTerminator())) {
          auto yieledValue = yieldOpInFor.getResults()[posInArg - 1];
          yiledExpr = "<-yd" + getYieldVerify(yieledValue);
        }
      }
      else{ // If it is induction variable.TODO: iv can be changed by some branches.
        llvm::errs()<<"iv!"<<"\n";
        auto ivInitValue = forOp.getLowerBound();
        ivExpr = "("+ getSettledVerify(ivInitValue)+")";
      }
    }
    return std::string("Index" + parentOp + std::to_string(posInArg) +ivExpr+
                       iterExpr + yiledExpr);
  }

  return std::string("unknow");
}

#endif // GET_SETTLED_H