#include "pgomlir/Passes/Passes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace pgomlir;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"),
                                          llvm::cl::value_desc("filename"));
static cl::opt<std::string> outputFilename("o",
                                           llvm::cl::desc("Output filename"),
                                           llvm::cl::value_desc("filename"),
                                           llvm::cl::init("-"));
static cl::opt<bool>
    SettledAttrToSCFPass("settled-attr-to-scf-pass", cl::init(false),
                         cl::desc("Turn on settled-attr-to-scf-pass"));

static cl::opt<bool>
    BranchProbabilityInfoPass("branch-prob-info-pass", cl::init(false),
                              cl::desc("Turn on branch-prob-info-pass"));

static cl::opt<bool> SCFToCFPass("scf-to-cf", cl::init(false),
                                 cl::desc("SCF to CF with extra information"));

static cl::opt<bool>
    CFToLLVMWithAttrPass("cf-to-llvm-with-attr", cl::init(false),
                         cl::desc("CF to LLVM with extra information"));

static cl::opt<bool> ConvertFuncToLLVMPass(
    "convert-func-to-llvm",
    cl::desc("Convert a function to LLVM IR format using MLIR"),
    cl::init(false));

static cl::opt<bool> ConvertSCFToLLVMWithAttr(
  "convert-scf-to-llvm-with-attr",
    cl::desc("Convert SCF to LLVM IR format with attribute using MLIR"),
    cl::init(false));

static cl::opt<bool> AddMetadataToBlocksPass(
  "add-attr-as-metadata-to-blocks",
    cl::desc("Add attribute in loop region as llvm.metadata"),
    cl::init(false));


int main(int argc, char **argv) {
  // Register all MLIR dialects and passes.

  mlir::registerAllPasses();

  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);

  MLIRContext context(registry);

  mlir::PassManager pm(&context);

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  // Parse command line arguments.
  cl::ParseCommandLineOptions(argc, argv, "\n");

  llvm::errs() << inputFilename << "!\n";
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  llvm::errs() << "Loaded\n";
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;

  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error loading input file\n";
    return 1;
  }

  // Add the custom attribute pass to the pass manager.
  if (SettledAttrToSCFPass) {
    pm.addPass(mlir::pgomlir::createSettledAttrToSCFPass());

    // Run the pass on the module.
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }

  if (BranchProbabilityInfoPass) {
    pm.addPass(mlir::pgomlir::createSettledAttrToSCFPass());
    pm.addPass(mlir::pgomlir::createBranchProbabilityInfoPass());

    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }

  if (SCFToCFPass) {
    pm.addPass(mlir::pgomlir::createSCFToCFPass());

    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }
  if (CFToLLVMWithAttrPass) {
    pm.addPass(mlir::pgomlir::createCFToLLVMWithAttrPass());

    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }
  if (ConvertFuncToLLVMPass) {
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }
  if(ConvertSCFToLLVMWithAttr){
    pm.addPass(mlir::pgomlir::createSettledAttrToSCFPass());
    pm.addPass(mlir::pgomlir::createSCFToCFPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    //pm.addPass(mlir::pgomlir::createAddMetadataToBlocksPass());
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }
  if(AddMetadataToBlocksPass){
    llvm::errs()<<"Metad\n";
    pm.addPass(mlir::pgomlir::createAddMetadataToBlocksPass());
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }

  // Write the output file.
  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "Error opening output file: " << error.message() << "\n";
    return 1;
  }
  module->print(output);

  return 0;
}