/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace nvgpu {

class NVGPUDialect : public ::mlir::Dialect {
  explicit NVGPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~NVGPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("nvgpu");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

    /// Return true if the given MemRefType has an integer address
    /// space that matches the NVVM shared memory address space or
    /// is a gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool hasSharedMemoryAddressSpace(MemRefType type);

    /// Defines the MemRef memory space attribute numeric value that indicates
    /// a memref is located in global memory. This should correspond to the
    /// value used in NVVM.
    static constexpr unsigned kGlobaldMemoryAddressSpace = 1;

    /// Defines the MemRef memory space attribute numeric value that indicates
    /// a memref is located in shared memory. This should correspond to the
    /// value used in NVVM.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
  };
} // namespace nvgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::NVGPUDialect)
