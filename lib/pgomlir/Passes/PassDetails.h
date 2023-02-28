#ifndef PGOMLIR_TRANSFORMS_PASSDETAILS_H
#define PGOMLIR_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"

#include "pgomlir/Passes/Passes.h"

namespace mlir {
namespace pgomlir {

#define GEN_PASS_CLASSES
#include "pgomlir/Passes/Passes.h.inc"

} // namespace pgomlir
} // namespace mlir

#endif // PGOMLIR_TRANSFORMS_PASSDETAILS_H
