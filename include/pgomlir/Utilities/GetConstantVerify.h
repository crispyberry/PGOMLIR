#ifndef GET_CONSTANT_H
#define GET_CONSTANT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

std::string getYieldVerify(Value value) {
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
      return getYieldVerify(left) + "+" + getYieldVerify(right);
    } // Further we should deal with other binary op like AddFOP DivOP ...
    else if (auto selectOp =
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

      return "{"+lhsvalue + " "+predicate+" "+rhsvalue+"?"+truevalue + ":" + falsevalue+"}";
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
      if (posInArg == 1 || posInArg == 2) { // if it is in iter_args.
        auto iterInitValue = forOp.getInitArgs()[posInArg - 1];
        iterExpr = "->[" + getYieldVerify(iterInitValue) + "]";
        // scf.for binds iter_args with scf.yield, so we must take scf.yield
        // into consideration.
      }
    }
    return std::string("Index" + parentOp + llvm::utostr(posInArg) + iterExpr +
                       yiledExpr);
  }

  return std::string("unknow");
}

std::string getConstantVerify(Value value) {
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
      return getConstantVerify(left) + "+" + getConstantVerify(right);
    } // Further we should deal with other binary op like AddFOP DivOP ...
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
      if (posInArg == 1 || posInArg == 2) { // if it is in iter_args.
        auto iterInitValue = forOp.getInitArgs()[posInArg - 1];
        iterExpr = "[" + getConstantVerify(iterInitValue)+"]";
        // scf.for binds iter_args with scf.yield, so we must take scf.yield
        // into consideration.
        if (auto yieldOpInFor = dyn_cast_or_null<scf::YieldOp>(
                forOp.getLoopBody().back().getTerminator())) {
          auto yieledValue = yieldOpInFor.getResults()[posInArg - 1];
          yiledExpr = "->" + getYieldVerify(yieledValue);
        }
      }
    }
    return std::string("Index" + parentOp + llvm::utostr(posInArg) + iterExpr +
                       yiledExpr);
  }

  return std::string("unknow");
}

#endif // GET_CONSTANT_H