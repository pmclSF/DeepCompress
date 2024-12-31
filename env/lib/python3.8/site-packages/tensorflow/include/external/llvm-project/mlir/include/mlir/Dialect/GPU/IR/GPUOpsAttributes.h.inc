/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace gpu {
class GPUBlockMappingAttr;
class GPULinearIdMappingAttr;
class GPUMemorySpaceMappingAttr;
class GPUThreadMappingAttr;
class GPUWarpMappingAttr;
class AddressSpaceAttr;
class AllReduceOperationAttr;
class DimensionAttr;
class ShuffleModeAttr;
class MMAElementwiseOpAttr;
class ParallelLoopDimMappingAttr;
namespace detail {
struct GPUBlockMappingAttrStorage;
} // namespace detail
class GPUBlockMappingAttr : public ::mlir::Attribute::AttrBase<GPUBlockMappingAttr, ::mlir::Attribute, detail::GPUBlockMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static GPUBlockMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Blocks block);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"block"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Blocks getBlock() const;
  int64_t getMappingId() const;
};
namespace detail {
struct GPULinearIdMappingAttrStorage;
} // namespace detail
class GPULinearIdMappingAttr : public ::mlir::Attribute::AttrBase<GPULinearIdMappingAttr, ::mlir::Attribute, detail::GPULinearIdMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static GPULinearIdMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::LinearId linear_id);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"linear"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::LinearId getLinearId() const;
  int64_t getMappingId() const;
};
namespace detail {
struct GPUMemorySpaceMappingAttrStorage;
} // namespace detail
class GPUMemorySpaceMappingAttr : public ::mlir::Attribute::AttrBase<GPUMemorySpaceMappingAttr, ::mlir::Attribute, detail::GPUMemorySpaceMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static GPUMemorySpaceMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace address_space);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AddressSpace getAddressSpace() const;
  int64_t getMappingId() const;
};
namespace detail {
struct GPUThreadMappingAttrStorage;
} // namespace detail
class GPUThreadMappingAttr : public ::mlir::Attribute::AttrBase<GPUThreadMappingAttr, ::mlir::Attribute, detail::GPUThreadMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static GPUThreadMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Threads thread);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"thread"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Threads getThread() const;
  int64_t getMappingId() const;
};
namespace detail {
struct GPUWarpMappingAttrStorage;
} // namespace detail
class GPUWarpMappingAttr : public ::mlir::Attribute::AttrBase<GPUWarpMappingAttr, ::mlir::Attribute, detail::GPUWarpMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static GPUWarpMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Warps warp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"warp"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Warps getWarp() const;
  int64_t getMappingId() const;
};
namespace detail {
struct AddressSpaceAttrStorage;
} // namespace detail
class AddressSpaceAttr : public ::mlir::Attribute::AttrBase<AddressSpaceAttr, ::mlir::Attribute, detail::AddressSpaceAttrStorage> {
public:
  using Base::Base;
  static AddressSpaceAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"address_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AddressSpace getValue() const;
};
namespace detail {
struct AllReduceOperationAttrStorage;
} // namespace detail
class AllReduceOperationAttr : public ::mlir::Attribute::AttrBase<AllReduceOperationAttr, ::mlir::Attribute, detail::AllReduceOperationAttrStorage> {
public:
  using Base::Base;
  static AllReduceOperationAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AllReduceOperation value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"all_reduce_op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AllReduceOperation getValue() const;
};
namespace detail {
struct DimensionAttrStorage;
} // namespace detail
class DimensionAttr : public ::mlir::Attribute::AttrBase<DimensionAttr, ::mlir::Attribute, detail::DimensionAttrStorage> {
public:
  using Base::Base;
  static DimensionAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Dimension value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dim"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Dimension getValue() const;
};
namespace detail {
struct ShuffleModeAttrStorage;
} // namespace detail
class ShuffleModeAttr : public ::mlir::Attribute::AttrBase<ShuffleModeAttr, ::mlir::Attribute, detail::ShuffleModeAttrStorage> {
public:
  using Base::Base;
  static ShuffleModeAttr get(::mlir::MLIRContext *context, ::mlir::gpu::ShuffleMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shuffle_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::ShuffleMode getValue() const;
};
namespace detail {
struct MMAElementwiseOpAttrStorage;
} // namespace detail
class MMAElementwiseOpAttr : public ::mlir::Attribute::AttrBase<MMAElementwiseOpAttr, ::mlir::Attribute, detail::MMAElementwiseOpAttrStorage> {
public:
  using Base::Base;
  static MMAElementwiseOpAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MMAElementwiseOp value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_element_wise"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MMAElementwiseOp getValue() const;
};
namespace detail {
struct ParallelLoopDimMappingAttrStorage;
} // namespace detail
class ParallelLoopDimMappingAttr : public ::mlir::Attribute::AttrBase<ParallelLoopDimMappingAttr, ::mlir::Attribute, detail::ParallelLoopDimMappingAttrStorage> {
public:
  using Base::Base;
  static ParallelLoopDimMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Processor processor, AffineMap map, AffineMap bound);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"loop_dim_map"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Processor getProcessor() const;
  AffineMap getMap() const;
  AffineMap getBound() const;
};
} // namespace gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUBlockMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPULinearIdMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUMemorySpaceMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUThreadMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUWarpMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AddressSpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ParallelLoopDimMappingAttr)

#endif  // GET_ATTRDEF_CLASSES

