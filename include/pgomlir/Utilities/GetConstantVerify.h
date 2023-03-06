#ifndef GET_CONSTANT_H
#define GET_CONSTANT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

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
    if (isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      parentOp = std::string("Func");
    } else if (isa<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      parentOp = std::string("For");
    }
    return std::string("blockArgIndex:" + parentOp +
                       llvm::utostr(blockArg.getArgNumber()));
  }

  return std::string("unknow");
}

#endif // GET_CONSTANT_H