#ifndef GET_CONSTANT_H
#define GET_CONSTANT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
std::string getConstantVerify(Value value) {
  std::string result;
  if (auto producer = value.getDefiningOp()) {
    if (auto cosntantIndexOp =
            llvm::dyn_cast_or_null<arith::ConstantIndexOp>(producer)) {
      result = std::to_string(cosntantIndexOp.value());
    } else if (auto cosntantIntOp =
                   llvm::dyn_cast_or_null<arith::ConstantIntOp>(producer)) {
      result = std::to_string(cosntantIntOp.value());
    } else {
      result = std::string("unknown");
    }
  } else {
    auto blockArg = value.cast<BlockArgument>();
    result =
        std::string("blockArgIndex:" + llvm::utostr(blockArg.getArgNumber()));
  }
  return result;
}

#endif // GET_CONSTANT_H