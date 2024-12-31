/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace NVVM {
class Barrier0Op;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockDimXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockDimYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockDimZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockIdXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockIdYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockIdZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class CpAsyncCommitGroupOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class CpAsyncOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class CpAsyncWaitGroupOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class GridDimXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class GridDimYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class GridDimZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class LaneIdOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class LdMatrixOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class MmaOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class RcpApproxFtzF32Op;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ReduxOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ShflOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class SyncWarpOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ThreadIdXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ThreadIdYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ThreadIdZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class VoteBallotOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WMMALoadOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WMMAMmaOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WMMAStoreOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WarpSizeOp;
} // namespace NVVM
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::Barrier0Op declarations
//===----------------------------------------------------------------------===//

namespace detail {
class Barrier0OpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  Barrier0OpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class Barrier0OpGenericAdaptor : public detail::Barrier0OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Barrier0OpGenericAdaptorBase;
public:
  Barrier0OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Barrier0OpAdaptor : public Barrier0OpGenericAdaptor<::mlir::ValueRange> {
public:
  using Barrier0OpGenericAdaptor::Barrier0OpGenericAdaptor;
  Barrier0OpAdaptor(Barrier0Op op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class Barrier0Op : public ::mlir::Op<Barrier0Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Barrier0OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Barrier0OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.barrier0");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::Barrier0Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimXOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BlockDimXOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BlockDimXOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BlockDimXOpGenericAdaptor : public detail::BlockDimXOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockDimXOpGenericAdaptorBase;
public:
  BlockDimXOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockDimXOpAdaptor : public BlockDimXOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockDimXOpGenericAdaptor::BlockDimXOpGenericAdaptor;
  BlockDimXOpAdaptor(BlockDimXOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BlockDimXOp : public ::mlir::Op<BlockDimXOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimXOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockDimXOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ntid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimYOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BlockDimYOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BlockDimYOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BlockDimYOpGenericAdaptor : public detail::BlockDimYOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockDimYOpGenericAdaptorBase;
public:
  BlockDimYOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockDimYOpAdaptor : public BlockDimYOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockDimYOpGenericAdaptor::BlockDimYOpGenericAdaptor;
  BlockDimYOpAdaptor(BlockDimYOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BlockDimYOp : public ::mlir::Op<BlockDimYOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimYOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockDimYOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ntid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimZOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BlockDimZOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BlockDimZOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BlockDimZOpGenericAdaptor : public detail::BlockDimZOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockDimZOpGenericAdaptorBase;
public:
  BlockDimZOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockDimZOpAdaptor : public BlockDimZOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockDimZOpGenericAdaptor::BlockDimZOpGenericAdaptor;
  BlockDimZOpAdaptor(BlockDimZOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BlockDimZOp : public ::mlir::Op<BlockDimZOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimZOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockDimZOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ntid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdXOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BlockIdXOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BlockIdXOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BlockIdXOpGenericAdaptor : public detail::BlockIdXOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockIdXOpGenericAdaptorBase;
public:
  BlockIdXOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockIdXOpAdaptor : public BlockIdXOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockIdXOpGenericAdaptor::BlockIdXOpGenericAdaptor;
  BlockIdXOpAdaptor(BlockIdXOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BlockIdXOp : public ::mlir::Op<BlockIdXOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdXOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockIdXOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ctaid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdYOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BlockIdYOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BlockIdYOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BlockIdYOpGenericAdaptor : public detail::BlockIdYOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockIdYOpGenericAdaptorBase;
public:
  BlockIdYOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockIdYOpAdaptor : public BlockIdYOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockIdYOpGenericAdaptor::BlockIdYOpGenericAdaptor;
  BlockIdYOpAdaptor(BlockIdYOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BlockIdYOp : public ::mlir::Op<BlockIdYOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdYOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockIdYOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ctaid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdZOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BlockIdZOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BlockIdZOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BlockIdZOpGenericAdaptor : public detail::BlockIdZOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockIdZOpGenericAdaptorBase;
public:
  BlockIdZOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockIdZOpAdaptor : public BlockIdZOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockIdZOpGenericAdaptor::BlockIdZOpGenericAdaptor;
  BlockIdZOpAdaptor(BlockIdZOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BlockIdZOp : public ::mlir::Op<BlockIdZOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdZOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockIdZOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ctaid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncCommitGroupOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CpAsyncCommitGroupOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CpAsyncCommitGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class CpAsyncCommitGroupOpGenericAdaptor : public detail::CpAsyncCommitGroupOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CpAsyncCommitGroupOpGenericAdaptorBase;
public:
  CpAsyncCommitGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CpAsyncCommitGroupOpAdaptor : public CpAsyncCommitGroupOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CpAsyncCommitGroupOpGenericAdaptor::CpAsyncCommitGroupOpGenericAdaptor;
  CpAsyncCommitGroupOpAdaptor(CpAsyncCommitGroupOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CpAsyncCommitGroupOp : public ::mlir::Op<CpAsyncCommitGroupOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CpAsyncCommitGroupOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CpAsyncCommitGroupOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.cp.async.commit.group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncCommitGroupOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CpAsyncOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CpAsyncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getSizeAttr();
  uint32_t getSize();
  ::mlir::UnitAttr getBypassL1Attr();
  ::std::optional<bool> getBypassL1();
};
} // namespace detail
template <typename RangeT>
class CpAsyncOpGenericAdaptor : public detail::CpAsyncOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CpAsyncOpGenericAdaptorBase;
public:
  CpAsyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDst() {
    return (*getODSOperands(0).begin());
  }

  ValueT getSrc() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CpAsyncOpAdaptor : public CpAsyncOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CpAsyncOpGenericAdaptor::CpAsyncOpGenericAdaptor;
  CpAsyncOpAdaptor(CpAsyncOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CpAsyncOp : public ::mlir::Op<CpAsyncOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CpAsyncOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CpAsyncOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("bypass_l1"), ::llvm::StringRef("size")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBypassL1AttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBypassL1AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.cp.async.shared.global");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getDst();
  ::mlir::Value getSrc();
  ::mlir::MutableOperandRange getDstMutable();
  ::mlir::MutableOperandRange getSrcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getSizeAttr();
  uint32_t getSize();
  ::mlir::UnitAttr getBypassL1Attr();
  ::std::optional<bool> getBypassL1();
  void setSizeAttr(::mlir::IntegerAttr attr);
  void setSize(uint32_t attrValue);
  void setBypassL1Attr(::mlir::UnitAttr attr);
  void setBypassL1(bool attrValue);
  ::mlir::Attribute removeBypassL1Attr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::IntegerAttr size, /*optional*/::mlir::UnitAttr bypass_l1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::IntegerAttr size, /*optional*/::mlir::UnitAttr bypass_l1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, uint32_t size, /*optional*/::mlir::UnitAttr bypass_l1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, uint32_t size, /*optional*/::mlir::UnitAttr bypass_l1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncWaitGroupOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CpAsyncWaitGroupOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CpAsyncWaitGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
};
} // namespace detail
template <typename RangeT>
class CpAsyncWaitGroupOpGenericAdaptor : public detail::CpAsyncWaitGroupOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CpAsyncWaitGroupOpGenericAdaptorBase;
public:
  CpAsyncWaitGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CpAsyncWaitGroupOpAdaptor : public CpAsyncWaitGroupOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CpAsyncWaitGroupOpGenericAdaptor::CpAsyncWaitGroupOpGenericAdaptor;
  CpAsyncWaitGroupOpAdaptor(CpAsyncWaitGroupOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CpAsyncWaitGroupOp : public ::mlir::Op<CpAsyncWaitGroupOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CpAsyncWaitGroupOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CpAsyncWaitGroupOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("n")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.cp.async.wait.group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  void setNAttr(::mlir::IntegerAttr attr);
  void setN(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr n);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr n);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t n);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t n);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncWaitGroupOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimXOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GridDimXOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GridDimXOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class GridDimXOpGenericAdaptor : public detail::GridDimXOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GridDimXOpGenericAdaptorBase;
public:
  GridDimXOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GridDimXOpAdaptor : public GridDimXOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GridDimXOpGenericAdaptor::GridDimXOpGenericAdaptor;
  GridDimXOpAdaptor(GridDimXOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GridDimXOp : public ::mlir::Op<GridDimXOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimXOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GridDimXOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.nctaid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimYOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GridDimYOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GridDimYOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class GridDimYOpGenericAdaptor : public detail::GridDimYOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GridDimYOpGenericAdaptorBase;
public:
  GridDimYOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GridDimYOpAdaptor : public GridDimYOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GridDimYOpGenericAdaptor::GridDimYOpGenericAdaptor;
  GridDimYOpAdaptor(GridDimYOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GridDimYOp : public ::mlir::Op<GridDimYOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimYOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GridDimYOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.nctaid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimZOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GridDimZOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GridDimZOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class GridDimZOpGenericAdaptor : public detail::GridDimZOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GridDimZOpGenericAdaptorBase;
public:
  GridDimZOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GridDimZOpAdaptor : public GridDimZOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GridDimZOpGenericAdaptor::GridDimZOpGenericAdaptor;
  GridDimZOpAdaptor(GridDimZOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GridDimZOp : public ::mlir::Op<GridDimZOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimZOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GridDimZOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.nctaid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::LaneIdOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class LaneIdOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  LaneIdOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class LaneIdOpGenericAdaptor : public detail::LaneIdOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LaneIdOpGenericAdaptorBase;
public:
  LaneIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LaneIdOpAdaptor : public LaneIdOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LaneIdOpGenericAdaptor::LaneIdOpGenericAdaptor;
  LaneIdOpAdaptor(LaneIdOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class LaneIdOp : public ::mlir::Op<LaneIdOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaneIdOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LaneIdOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.laneid");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::LaneIdOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::LdMatrixOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class LdMatrixOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  LdMatrixOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getNumAttr();
  uint32_t getNum();
  ::mlir::NVVM::MMALayoutAttr getLayoutAttr();
  ::mlir::NVVM::MMALayout getLayout();
};
} // namespace detail
template <typename RangeT>
class LdMatrixOpGenericAdaptor : public detail::LdMatrixOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LdMatrixOpGenericAdaptorBase;
public:
  LdMatrixOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LdMatrixOpAdaptor : public LdMatrixOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LdMatrixOpGenericAdaptor::LdMatrixOpGenericAdaptor;
  LdMatrixOpAdaptor(LdMatrixOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class LdMatrixOp : public ::mlir::Op<LdMatrixOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LdMatrixOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LdMatrixOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("layout"), ::llvm::StringRef("num")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLayoutAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.ldmatrix");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> getPtr();
  ::mlir::MutableOperandRange getPtrMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  ::mlir::IntegerAttr getNumAttr();
  uint32_t getNum();
  ::mlir::NVVM::MMALayoutAttr getLayoutAttr();
  ::mlir::NVVM::MMALayout getLayout();
  void setNumAttr(::mlir::IntegerAttr attr);
  void setNum(uint32_t attrValue);
  void setLayoutAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayout(::mlir::NVVM::MMALayout attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::IntegerAttr num, ::mlir::NVVM::MMALayoutAttr layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::IntegerAttr num, ::mlir::NVVM::MMALayoutAttr layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, uint32_t num, ::mlir::NVVM::MMALayout layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, uint32_t num, ::mlir::NVVM::MMALayout layout);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::LdMatrixOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::MmaOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MmaOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MmaOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::NVVM::MMAShapeAttr getShapeAttr();
  ::mlir::NVVM::MMAShapeAttr getShape();
  ::mlir::NVVM::MMAB1OpAttr getB1OpAttr();
  ::std::optional<::mlir::NVVM::MMAB1Op> getB1Op();
  ::mlir::NVVM::MMAIntOverflowAttr getIntOverflowBehaviorAttr();
  ::std::optional<::mlir::NVVM::MMAIntOverflow> getIntOverflowBehavior();
  ::mlir::NVVM::MMALayoutAttr getLayoutAAttr();
  ::mlir::NVVM::MMALayout getLayoutA();
  ::mlir::NVVM::MMALayoutAttr getLayoutBAttr();
  ::mlir::NVVM::MMALayout getLayoutB();
  ::mlir::NVVM::MMATypesAttr getMultiplicandAPtxTypeAttr();
  ::std::optional<::mlir::NVVM::MMATypes> getMultiplicandAPtxType();
  ::mlir::NVVM::MMATypesAttr getMultiplicandBPtxTypeAttr();
  ::std::optional<::mlir::NVVM::MMATypes> getMultiplicandBPtxType();
};
} // namespace detail
template <typename RangeT>
class MmaOpGenericAdaptor : public detail::MmaOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MmaOpGenericAdaptorBase;
public:
  MmaOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperandA() {
    return getODSOperands(0);
  }

  RangeT getOperandB() {
    return getODSOperands(1);
  }

  RangeT getOperandC() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MmaOpAdaptor : public MmaOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MmaOpGenericAdaptor::MmaOpGenericAdaptor;
  MmaOpAdaptor(MmaOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MmaOp : public ::mlir::Op<MmaOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MmaOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MmaOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("b1Op"), ::llvm::StringRef("intOverflowBehavior"), ::llvm::StringRef("layoutA"), ::llvm::StringRef("layoutB"), ::llvm::StringRef("multiplicandAPtxType"), ::llvm::StringRef("multiplicandBPtxType"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("shape")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getB1OpAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getB1OpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIntOverflowBehaviorAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIntOverflowBehaviorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLayoutAAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLayoutAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLayoutBAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLayoutBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getMultiplicandAPtxTypeAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getMultiplicandAPtxTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getMultiplicandBPtxTypeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getMultiplicandBPtxTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getShapeAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getShapeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.mma.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getOperandA();
  ::mlir::Operation::operand_range getOperandB();
  ::mlir::Operation::operand_range getOperandC();
  ::mlir::MutableOperandRange getOperandAMutable();
  ::mlir::MutableOperandRange getOperandBMutable();
  ::mlir::MutableOperandRange getOperandCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  ::mlir::NVVM::MMAShapeAttr getShapeAttr();
  ::mlir::NVVM::MMAShapeAttr getShape();
  ::mlir::NVVM::MMAB1OpAttr getB1OpAttr();
  ::std::optional<::mlir::NVVM::MMAB1Op> getB1Op();
  ::mlir::NVVM::MMAIntOverflowAttr getIntOverflowBehaviorAttr();
  ::std::optional<::mlir::NVVM::MMAIntOverflow> getIntOverflowBehavior();
  ::mlir::NVVM::MMALayoutAttr getLayoutAAttr();
  ::mlir::NVVM::MMALayout getLayoutA();
  ::mlir::NVVM::MMALayoutAttr getLayoutBAttr();
  ::mlir::NVVM::MMALayout getLayoutB();
  ::mlir::NVVM::MMATypesAttr getMultiplicandAPtxTypeAttr();
  ::std::optional<::mlir::NVVM::MMATypes> getMultiplicandAPtxType();
  ::mlir::NVVM::MMATypesAttr getMultiplicandBPtxTypeAttr();
  ::std::optional<::mlir::NVVM::MMATypes> getMultiplicandBPtxType();
  void setShapeAttr(::mlir::NVVM::MMAShapeAttr attr);
  void setB1OpAttr(::mlir::NVVM::MMAB1OpAttr attr);
  void setB1Op(::std::optional<::mlir::NVVM::MMAB1Op> attrValue);
  void setIntOverflowBehaviorAttr(::mlir::NVVM::MMAIntOverflowAttr attr);
  void setIntOverflowBehavior(::std::optional<::mlir::NVVM::MMAIntOverflow> attrValue);
  void setLayoutAAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayoutA(::mlir::NVVM::MMALayout attrValue);
  void setLayoutBAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayoutB(::mlir::NVVM::MMALayout attrValue);
  void setMultiplicandAPtxTypeAttr(::mlir::NVVM::MMATypesAttr attr);
  void setMultiplicandAPtxType(::std::optional<::mlir::NVVM::MMATypes> attrValue);
  void setMultiplicandBPtxTypeAttr(::mlir::NVVM::MMATypesAttr attr);
  void setMultiplicandBPtxType(::std::optional<::mlir::NVVM::MMATypes> attrValue);
  ::mlir::Attribute removeB1OpAttr();
  ::mlir::Attribute removeIntOverflowBehaviorAttr();
  ::mlir::Attribute removeMultiplicandAPtxTypeAttr();
  ::mlir::Attribute removeMultiplicandBPtxTypeAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operandA, ValueRange operandB, ValueRange operandC, ArrayRef<int64_t> shape, std::optional<MMAB1Op> b1Op, std::optional<MMAIntOverflow> intOverflow, std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes, std::optional<std::array<MMALayout, 2>> multiplicandLayouts);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
        static llvm::Intrinsic::ID getIntrinsicID(
              int64_t m, int64_t n, uint64_t k,
              std::optional<MMAB1Op> b1Op,
              std::optional<MMAIntOverflow> sat,
              mlir::NVVM::MMALayout layoutAEnum, mlir::NVVM::MMALayout layoutBEnum,
              mlir::NVVM::MMATypes eltypeAEnum, mlir::NVVM::MMATypes eltypeBEnum,
              mlir::NVVM::MMATypes eltypeCEnum, mlir::NVVM::MMATypes eltypeDEnum) {
          llvm::StringRef layoutA = stringifyEnum(layoutAEnum);
          llvm::StringRef layoutB = stringifyEnum(layoutBEnum);
          llvm::StringRef eltypeA = stringifyEnum(eltypeAEnum);
          llvm::StringRef eltypeB = stringifyEnum(eltypeBEnum);
          llvm::StringRef eltypeC = stringifyEnum(eltypeCEnum);
          llvm::StringRef eltypeD = stringifyEnum(eltypeDEnum);







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 4    && "tf32" == eltypeA && "tf32" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k4_row_col_tf32;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 8    && "tf32" == eltypeA && "tf32" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k8_row_col_tf32;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "bf16" == eltypeA && "bf16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_bf16;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 8    && "bf16" == eltypeA && "bf16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k8_row_col_bf16;







  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f64" == eltypeA && "f64" == eltypeB &&  "f64" == eltypeC &&  "f64" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f64;





  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_row_f16_f16;

  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f16_f16;

  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_col_row_f16_f16;

  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_col_col_f16_f16;

  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_row_f32_f16;

  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f16;

  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_col_row_f32_f16;

  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_col_col_f32_f16;









  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_row_f32_f32;

  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f32;

  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_col_row_f32_f32;

  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 8 && k == 4    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k4_col_col_f32_f32;



  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 8    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k8_row_col_f16_f16;























  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 8    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k8_row_col_f32_f32;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_f16_f16;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB &&  "f16" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_f32_f16;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f16" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_f16_f32;







  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB &&  "f32" == eltypeC &&  "f32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_f32_f32;







  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "s8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "s8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_satfinite_s8;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "s8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_s8_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "s8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_satfinite_s8_u8;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "u8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_u8_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "u8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_satfinite_u8_s8;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "u8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 16    && "u8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k16_row_col_satfinite_u8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "s8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "s8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_satfinite_s8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "s8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_s8_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "s8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_satfinite_s8_u8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "u8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_u8_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "u8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_satfinite_u8_s8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "u8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 16    && "u8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k16_row_col_satfinite_u8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_s8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_s8_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_s8_u8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_u8_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u8" == eltypeA && "s8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_u8_s8;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u8" == eltypeA && "u8" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_u8;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "s4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_s4;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "s4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_satfinite_s4;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "s4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_s4_u4;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "s4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_satfinite_s4_u4;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "u4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_u4_s4;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "u4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_satfinite_u4_s4;






  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "u4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_u4;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 32    && "u4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m8n8k32_row_col_satfinite_u4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_s4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_s4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_s4_u4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "s4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_s4_u4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_u4_s4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_u4_s4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_u4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 32    && "u4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k32_row_col_satfinite_u4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "s4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_s4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "s4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_satfinite_s4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "s4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_s4_u4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "s4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_satfinite_s4_u4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "u4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_u4_s4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "u4" == eltypeA && "s4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_satfinite_u4_s4;






  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "u4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_u4;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 64    && "u4" == eltypeA && "u4" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 1 == static_cast<int>(*sat) : true))
    return llvm::Intrinsic::nvvm_mma_m16n8k64_row_col_satfinite_u4;








  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 128    && "b1" == eltypeA && "b1" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true) && (b1Op.has_value() ? MMAB1Op::xor_popc == *b1Op : true))
    return llvm::Intrinsic::nvvm_mma_xor_popc_m8n8k128_row_col_b1;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 8 && k == 128    && "b1" == eltypeA && "b1" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true) && (b1Op.has_value() ? MMAB1Op::and_popc == *b1Op : true))
    return llvm::Intrinsic::nvvm_mma_and_popc_m8n8k128_row_col_b1;














  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 128    && "b1" == eltypeA && "b1" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true) && (b1Op.has_value() ? MMAB1Op::xor_popc == *b1Op : true))
    return llvm::Intrinsic::nvvm_mma_xor_popc_m16n8k128_row_col_b1;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 128    && "b1" == eltypeA && "b1" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true) && (b1Op.has_value() ? MMAB1Op::and_popc == *b1Op : true))
    return llvm::Intrinsic::nvvm_mma_and_popc_m16n8k128_row_col_b1;














  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 256    && "b1" == eltypeA && "b1" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true) && (b1Op.has_value() ? MMAB1Op::xor_popc == *b1Op : true))
    return llvm::Intrinsic::nvvm_mma_xor_popc_m16n8k256_row_col_b1;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 8 && k == 256    && "b1" == eltypeA && "b1" == eltypeB &&  "s32" == eltypeC &&  "s32" == eltypeD  && (sat.has_value()  ? 0 == static_cast<int>(*sat) : true) && (b1Op.has_value() ? MMAB1Op::and_popc == *b1Op : true))
    return llvm::Intrinsic::nvvm_mma_and_popc_m16n8k256_row_col_b1;










          return 0;
        }

        static std::optional<mlir::NVVM::MMATypes> inferOperandMMAType(Type operandElType,
          bool isAccumulator);

        MMATypes accumPtxType();
        MMATypes resultPtxType();
      };
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::RcpApproxFtzF32Op declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RcpApproxFtzF32OpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RcpApproxFtzF32OpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class RcpApproxFtzF32OpGenericAdaptor : public detail::RcpApproxFtzF32OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RcpApproxFtzF32OpGenericAdaptorBase;
public:
  RcpApproxFtzF32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getArg() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RcpApproxFtzF32OpAdaptor : public RcpApproxFtzF32OpGenericAdaptor<::mlir::ValueRange> {
public:
  using RcpApproxFtzF32OpGenericAdaptor::RcpApproxFtzF32OpGenericAdaptor;
  RcpApproxFtzF32OpAdaptor(RcpApproxFtzF32Op op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RcpApproxFtzF32Op : public ::mlir::Op<RcpApproxFtzF32Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::FloatType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RcpApproxFtzF32OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RcpApproxFtzF32OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.rcp.approx.ftz.f");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::FloatType> getArg();
  ::mlir::MutableOperandRange getArgMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::FloatType> getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::RcpApproxFtzF32Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ReduxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReduxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReduxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::NVVM::ReduxKindAttr getKindAttr();
  ::mlir::NVVM::ReduxKind getKind();
};
} // namespace detail
template <typename RangeT>
class ReduxOpGenericAdaptor : public detail::ReduxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReduxOpGenericAdaptorBase;
public:
  ReduxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVal() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMaskAndClamp() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReduxOpAdaptor : public ReduxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReduxOpGenericAdaptor::ReduxOpGenericAdaptor;
  ReduxOpAdaptor(ReduxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReduxOp : public ::mlir::Op<ReduxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReduxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getKindAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getKindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.redux.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getVal();
  ::mlir::TypedValue<::mlir::IntegerType> getMaskAndClamp();
  ::mlir::MutableOperandRange getValMutable();
  ::mlir::MutableOperandRange getMaskAndClampMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  ::mlir::NVVM::ReduxKindAttr getKindAttr();
  ::mlir::NVVM::ReduxKind getKind();
  void setKindAttr(::mlir::NVVM::ReduxKindAttr attr);
  void setKind(::mlir::NVVM::ReduxKind attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val, ::mlir::NVVM::ReduxKindAttr kind, ::mlir::Value mask_and_clamp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::NVVM::ReduxKindAttr kind, ::mlir::Value mask_and_clamp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val, ::mlir::NVVM::ReduxKind kind, ::mlir::Value mask_and_clamp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::NVVM::ReduxKind kind, ::mlir::Value mask_and_clamp);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ReduxOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ShflOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ShflOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ShflOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::NVVM::ShflKindAttr getKindAttr();
  ::mlir::NVVM::ShflKind getKind();
  ::mlir::UnitAttr getReturnValueAndIsValidAttr();
  ::std::optional<bool> getReturnValueAndIsValid();
};
} // namespace detail
template <typename RangeT>
class ShflOpGenericAdaptor : public detail::ShflOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ShflOpGenericAdaptorBase;
public:
  ShflOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDst() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVal() {
    return (*getODSOperands(1).begin());
  }

  ValueT getOffset() {
    return (*getODSOperands(2).begin());
  }

  ValueT getMaskAndClamp() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ShflOpAdaptor : public ShflOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ShflOpGenericAdaptor::ShflOpGenericAdaptor;
  ShflOpAdaptor(ShflOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ShflOp : public ::mlir::Op<ShflOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShflOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ShflOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind"), ::llvm::StringRef("return_value_and_is_valid")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getKindAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getKindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getReturnValueAndIsValidAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getReturnValueAndIsValidAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.shfl.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getDst();
  ::mlir::Value getVal();
  ::mlir::TypedValue<::mlir::IntegerType> getOffset();
  ::mlir::TypedValue<::mlir::IntegerType> getMaskAndClamp();
  ::mlir::MutableOperandRange getDstMutable();
  ::mlir::MutableOperandRange getValMutable();
  ::mlir::MutableOperandRange getOffsetMutable();
  ::mlir::MutableOperandRange getMaskAndClampMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  ::mlir::NVVM::ShflKindAttr getKindAttr();
  ::mlir::NVVM::ShflKind getKind();
  ::mlir::UnitAttr getReturnValueAndIsValidAttr();
  ::std::optional<bool> getReturnValueAndIsValid();
  void setKindAttr(::mlir::NVVM::ShflKindAttr attr);
  void setKind(::mlir::NVVM::ShflKind attrValue);
  void setReturnValueAndIsValidAttr(::mlir::UnitAttr attr);
  void setReturnValueAndIsValid(bool attrValue);
  ::mlir::Attribute removeReturnValueAndIsValidAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKindAttr kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKindAttr kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKind kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKind kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::SyncWarpOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SyncWarpOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SyncWarpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class SyncWarpOpGenericAdaptor : public detail::SyncWarpOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SyncWarpOpGenericAdaptorBase;
public:
  SyncWarpOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMask() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SyncWarpOpAdaptor : public SyncWarpOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SyncWarpOpGenericAdaptor::SyncWarpOpGenericAdaptor;
  SyncWarpOpAdaptor(SyncWarpOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SyncWarpOp : public ::mlir::Op<SyncWarpOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SyncWarpOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SyncWarpOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.bar.warp.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getMask();
  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value mask);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::SyncWarpOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdXOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ThreadIdXOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ThreadIdXOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ThreadIdXOpGenericAdaptor : public detail::ThreadIdXOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ThreadIdXOpGenericAdaptorBase;
public:
  ThreadIdXOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ThreadIdXOpAdaptor : public ThreadIdXOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ThreadIdXOpGenericAdaptor::ThreadIdXOpGenericAdaptor;
  ThreadIdXOpAdaptor(ThreadIdXOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ThreadIdXOp : public ::mlir::Op<ThreadIdXOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdXOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ThreadIdXOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.tid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdYOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ThreadIdYOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ThreadIdYOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ThreadIdYOpGenericAdaptor : public detail::ThreadIdYOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ThreadIdYOpGenericAdaptorBase;
public:
  ThreadIdYOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ThreadIdYOpAdaptor : public ThreadIdYOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ThreadIdYOpGenericAdaptor::ThreadIdYOpGenericAdaptor;
  ThreadIdYOpAdaptor(ThreadIdYOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ThreadIdYOp : public ::mlir::Op<ThreadIdYOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdYOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ThreadIdYOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.tid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdZOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ThreadIdZOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ThreadIdZOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ThreadIdZOpGenericAdaptor : public detail::ThreadIdZOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ThreadIdZOpGenericAdaptorBase;
public:
  ThreadIdZOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ThreadIdZOpAdaptor : public ThreadIdZOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ThreadIdZOpGenericAdaptor::ThreadIdZOpGenericAdaptor;
  ThreadIdZOpAdaptor(ThreadIdZOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ThreadIdZOp : public ::mlir::Op<ThreadIdZOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdZOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ThreadIdZOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.tid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::VoteBallotOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class VoteBallotOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  VoteBallotOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class VoteBallotOpGenericAdaptor : public detail::VoteBallotOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::VoteBallotOpGenericAdaptorBase;
public:
  VoteBallotOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMask() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPred() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class VoteBallotOpAdaptor : public VoteBallotOpGenericAdaptor<::mlir::ValueRange> {
public:
  using VoteBallotOpGenericAdaptor::VoteBallotOpGenericAdaptor;
  VoteBallotOpAdaptor(VoteBallotOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class VoteBallotOp : public ::mlir::Op<VoteBallotOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VoteBallotOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = VoteBallotOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.vote.ballot.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getMask();
  ::mlir::Value getPred();
  ::mlir::MutableOperandRange getMaskMutable();
  ::mlir::MutableOperandRange getPredMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::VoteBallotOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WMMALoadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WMMALoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::NVVM::MMALayoutAttr getLayoutAttr();
  ::mlir::NVVM::MMALayout getLayout();
  ::mlir::NVVM::MMATypesAttr getEltypeAttr();
  ::mlir::NVVM::MMATypes getEltype();
  ::mlir::NVVM::MMAFragAttr getFragAttr();
  ::mlir::NVVM::MMAFrag getFrag();
};
} // namespace detail
template <typename RangeT>
class WMMALoadOpGenericAdaptor : public detail::WMMALoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WMMALoadOpGenericAdaptorBase;
public:
  WMMALoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getStride() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WMMALoadOpAdaptor : public WMMALoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WMMALoadOpGenericAdaptor::WMMALoadOpGenericAdaptor;
  WMMALoadOpAdaptor(WMMALoadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WMMALoadOp : public ::mlir::Op<WMMALoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WMMALoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WMMALoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("eltype"), ::llvm::StringRef("frag"), ::llvm::StringRef("k"), ::llvm::StringRef("layout"), ::llvm::StringRef("m"), ::llvm::StringRef("n")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getEltypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getEltypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getFragAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getFragAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getKAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getKAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLayoutAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getMAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getMAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.wmma.load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> getPtr();
  ::mlir::TypedValue<::mlir::IntegerType> getStride();
  ::mlir::MutableOperandRange getPtrMutable();
  ::mlir::MutableOperandRange getStrideMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::NVVM::MMALayoutAttr getLayoutAttr();
  ::mlir::NVVM::MMALayout getLayout();
  ::mlir::NVVM::MMATypesAttr getEltypeAttr();
  ::mlir::NVVM::MMATypes getEltype();
  ::mlir::NVVM::MMAFragAttr getFragAttr();
  ::mlir::NVVM::MMAFrag getFrag();
  void setMAttr(::mlir::IntegerAttr attr);
  void setM(uint32_t attrValue);
  void setNAttr(::mlir::IntegerAttr attr);
  void setN(uint32_t attrValue);
  void setKAttr(::mlir::IntegerAttr attr);
  void setK(uint32_t attrValue);
  void setLayoutAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayout(::mlir::NVVM::MMALayout attrValue);
  void setEltypeAttr(::mlir::NVVM::MMATypesAttr attr);
  void setEltype(::mlir::NVVM::MMATypes attrValue);
  void setFragAttr(::mlir::NVVM::MMAFragAttr attr);
  void setFrag(::mlir::NVVM::MMAFrag attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::NVVM::MMAFragAttr frag);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::NVVM::MMAFragAttr frag);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::NVVM::MMAFrag frag);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::NVVM::MMAFrag frag);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static llvm::Intrinsic::ID getIntrinsicID(int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,mlir::NVVM::MMATypes eltypeEnum,mlir::NVVM::MMAFrag fragEnum) {llvm::StringRef layout = stringifyEnum(layoutEnum);llvm::StringRef eltype = stringifyEnum(eltypeEnum);llvm::StringRef frag = stringifyEnum(fragEnum);

  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "s8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "s8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "u8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "u8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "s8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "s8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "u8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "u8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "s8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "s8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "u8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "u8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "s8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "s8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "u8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "u8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "s8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "s8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "u8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "u8" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "s8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "s8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "u8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "u8" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "s32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "s32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "s32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "s32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "s32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "s32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_col_stride;
  return 0;}
  /// Helpers to find valid n dimension based on mxk load shape.
  static int inferNDimension(int m, int k, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (m == 16 && k == 16 && "f16" == eltype)  return 16;
  if (m == 16 && k == 16 && "s8" == eltype)  return 16;
  if (m == 16 && k == 16 && "u8" == eltype)  return 16;
  if (m == 32 && k == 16 && "f16" == eltype)  return 8;
  if (m == 32 && k == 16 && "s8" == eltype)  return 8;
  if (m == 32 && k == 16 && "u8" == eltype)  return 8;
  if (m == 8 && k == 16 && "f16" == eltype)  return 32;
  if (m == 8 && k == 16 && "s8" == eltype)  return 32;
  if (m == 8 && k == 16 && "u8" == eltype)  return 32;
  if (m == 16 && k == 8 && "tf32" == eltype)  return 16;
  return 0;}
  /// Helpers to find valid m dimension based on kxn load shape.
  static int inferMDimension(int k, int n, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (n == 16 && k == 16 && "f16" == eltype)  return 16;
  if (n == 16 && k == 16 && "s8" == eltype)  return 16;
  if (n == 16 && k == 16 && "u8" == eltype)  return 16;
  if (n == 8 && k == 16 && "f16" == eltype)  return 32;
  if (n == 8 && k == 16 && "s8" == eltype)  return 32;
  if (n == 8 && k == 16 && "u8" == eltype)  return 32;
  if (n == 32 && k == 16 && "f16" == eltype)  return 8;
  if (n == 32 && k == 16 && "s8" == eltype)  return 8;
  if (n == 32 && k == 16 && "u8" == eltype)  return 8;
  if (n == 16 && k == 8 && "tf32" == eltype)  return 16;
  return 0;}
  /// Helpers to find valid k dimension based on mxn load shape.
  static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (m == 16 && n == 16 && "f16" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 16;
  if (m == 16 && n == 16 && "s32" == eltype)  return 16;
  if (m == 32 && n == 8 && "f16" == eltype)  return 16;
  if (m == 32 && n == 8 && "f32" == eltype)  return 16;
  if (m == 32 && n == 8 && "s32" == eltype)  return 16;
  if (m == 8 && n == 32 && "f16" == eltype)  return 16;
  if (m == 8 && n == 32 && "f32" == eltype)  return 16;
  if (m == 8 && n == 32 && "s32" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 8;
  return 0;}
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAMmaOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WMMAMmaOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WMMAMmaOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::NVVM::MMALayoutAttr getLayoutAAttr();
  ::mlir::NVVM::MMALayout getLayoutA();
  ::mlir::NVVM::MMALayoutAttr getLayoutBAttr();
  ::mlir::NVVM::MMALayout getLayoutB();
  ::mlir::NVVM::MMATypesAttr getEltypeAAttr();
  ::mlir::NVVM::MMATypes getEltypeA();
  ::mlir::NVVM::MMATypesAttr getEltypeBAttr();
  ::mlir::NVVM::MMATypes getEltypeB();
};
} // namespace detail
template <typename RangeT>
class WMMAMmaOpGenericAdaptor : public detail::WMMAMmaOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WMMAMmaOpGenericAdaptorBase;
public:
  WMMAMmaOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getArgs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WMMAMmaOpAdaptor : public WMMAMmaOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WMMAMmaOpGenericAdaptor::WMMAMmaOpGenericAdaptor;
  WMMAMmaOpAdaptor(WMMAMmaOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WMMAMmaOp : public ::mlir::Op<WMMAMmaOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WMMAMmaOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WMMAMmaOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("eltypeA"), ::llvm::StringRef("eltypeB"), ::llvm::StringRef("k"), ::llvm::StringRef("layoutA"), ::llvm::StringRef("layoutB"), ::llvm::StringRef("m"), ::llvm::StringRef("n")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getEltypeAAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getEltypeAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getEltypeBAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getEltypeBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getKAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getKAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLayoutAAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLayoutAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLayoutBAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLayoutBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getMAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getMAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.wmma.mma");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::NVVM::MMALayoutAttr getLayoutAAttr();
  ::mlir::NVVM::MMALayout getLayoutA();
  ::mlir::NVVM::MMALayoutAttr getLayoutBAttr();
  ::mlir::NVVM::MMALayout getLayoutB();
  ::mlir::NVVM::MMATypesAttr getEltypeAAttr();
  ::mlir::NVVM::MMATypes getEltypeA();
  ::mlir::NVVM::MMATypesAttr getEltypeBAttr();
  ::mlir::NVVM::MMATypes getEltypeB();
  void setMAttr(::mlir::IntegerAttr attr);
  void setM(uint32_t attrValue);
  void setNAttr(::mlir::IntegerAttr attr);
  void setN(uint32_t attrValue);
  void setKAttr(::mlir::IntegerAttr attr);
  void setK(uint32_t attrValue);
  void setLayoutAAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayoutA(::mlir::NVVM::MMALayout attrValue);
  void setLayoutBAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayoutB(::mlir::NVVM::MMALayout attrValue);
  void setEltypeAAttr(::mlir::NVVM::MMATypesAttr attr);
  void setEltypeA(::mlir::NVVM::MMATypes attrValue);
  void setEltypeBAttr(::mlir::NVVM::MMATypesAttr attr);
  void setEltypeB(::mlir::NVVM::MMATypes attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, ::mlir::NVVM::MMATypesAttr eltypeA, ::mlir::NVVM::MMATypesAttr eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, ::mlir::NVVM::MMATypesAttr eltypeA, ::mlir::NVVM::MMATypesAttr eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, ::mlir::NVVM::MMATypes eltypeA, ::mlir::NVVM::MMATypes eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, ::mlir::NVVM::MMATypes eltypeA, ::mlir::NVVM::MMATypes eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static llvm::Intrinsic::ID getIntrinsicID(int m, int n, int k, mlir::NVVM::MMALayout layoutAEnum,mlir::NVVM::MMALayout layoutBEnum, mlir::NVVM::MMATypes eltypeAEnum,mlir::NVVM::MMATypes eltypeBEnum) {llvm::StringRef layoutA = stringifyEnum(layoutAEnum);llvm::StringRef layoutB = stringifyEnum(layoutBEnum);llvm::StringRef eltypeA = stringifyEnum(eltypeAEnum);llvm::StringRef eltypeB = stringifyEnum(eltypeBEnum);


  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_row_row_tf32;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_row_col_tf32;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_col_row_tf32;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_col_col_tf32;
  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f16;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f16_f16;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f16_f16;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f16_f16;
  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f32_f32;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f32_f32;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f32_f32;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f32_f32;
  if (layoutA == "row" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f16_f16;
  if (layoutA == "row" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f16_f16;
  if (layoutA == "col" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f16_f16;
  if (layoutA == "col" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f16_f16;
  if (layoutA == "row" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f32_f32;
  if (layoutA == "row" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f32_f32;
  if (layoutA == "col" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f32_f32;
  if (layoutA == "col" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f32_f32;
  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f16_f16;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f16_f16;
  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f16_f16;
  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f16_f16;
  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f32_f32;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f32_f32;
  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f32_f32;
  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f32_f32;
  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_s8;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_s8;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_s8;
  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_u8;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_u8;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_u8;
  if (layoutA == "row" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_s8;
  if (layoutA == "col" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_s8;
  if (layoutA == "col" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_s8;
  if (layoutA == "row" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_u8;
  if (layoutA == "col" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_u8;
  if (layoutA == "col" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_u8;
  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_s8;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_s8;
  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_s8;
  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "s8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_s8;
  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_u8;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_u8;
  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_u8;
  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "u8" == eltypeA && "s32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_u8;
  return 0;}};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAMmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WMMAStoreOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WMMAStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::NVVM::MMALayoutAttr getLayoutAttr();
  ::mlir::NVVM::MMALayout getLayout();
  ::mlir::NVVM::MMATypesAttr getEltypeAttr();
  ::mlir::NVVM::MMATypes getEltype();
};
} // namespace detail
template <typename RangeT>
class WMMAStoreOpGenericAdaptor : public detail::WMMAStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WMMAStoreOpGenericAdaptorBase;
public:
  WMMAStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  RangeT getArgs() {
    return getODSOperands(1);
  }

  ValueT getStride() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WMMAStoreOpAdaptor : public WMMAStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WMMAStoreOpGenericAdaptor::WMMAStoreOpGenericAdaptor;
  WMMAStoreOpAdaptor(WMMAStoreOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WMMAStoreOp : public ::mlir::Op<WMMAStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WMMAStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WMMAStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("eltype"), ::llvm::StringRef("k"), ::llvm::StringRef("layout"), ::llvm::StringRef("m"), ::llvm::StringRef("n")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getEltypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getEltypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getKAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getKAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLayoutAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getMAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getMAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.wmma.store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> getPtr();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::TypedValue<::mlir::IntegerType> getStride();
  ::mlir::MutableOperandRange getPtrMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  ::mlir::MutableOperandRange getStrideMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::NVVM::MMALayoutAttr getLayoutAttr();
  ::mlir::NVVM::MMALayout getLayout();
  ::mlir::NVVM::MMATypesAttr getEltypeAttr();
  ::mlir::NVVM::MMATypes getEltype();
  void setMAttr(::mlir::IntegerAttr attr);
  void setM(uint32_t attrValue);
  void setNAttr(::mlir::IntegerAttr attr);
  void setN(uint32_t attrValue);
  void setKAttr(::mlir::IntegerAttr attr);
  void setK(uint32_t attrValue);
  void setLayoutAttr(::mlir::NVVM::MMALayoutAttr attr);
  void setLayout(::mlir::NVVM::MMALayout attrValue);
  void setEltypeAttr(::mlir::NVVM::MMATypesAttr attr);
  void setEltype(::mlir::NVVM::MMATypes attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static llvm::Intrinsic::ID getIntrinsicID(int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef layout = stringifyEnum(layoutEnum);  llvm::StringRef eltype = stringifyEnum(eltypeEnum);

  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "s32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_s32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "s32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_s32_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "s32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_s32_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "s32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_s32_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "s32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_s32_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "s32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_s32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_col_stride;
  return 0;}
  /// Helpers to find valid k dimension based on mxn store shape.
  static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (m == 16 && n == 16 && "f16" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 16;
  if (m == 16 && n == 16 && "s32" == eltype)  return 16;
  if (m == 32 && n == 8 && "f16" == eltype)  return 16;
  if (m == 32 && n == 8 && "f32" == eltype)  return 16;
  if (m == 32 && n == 8 && "s32" == eltype)  return 16;
  if (m == 8 && n == 32 && "f16" == eltype)  return 16;
  if (m == 8 && n == 32 && "f32" == eltype)  return 16;
  if (m == 8 && n == 32 && "s32" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 8;
  return 0;}};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAStoreOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WarpSizeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WarpSizeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WarpSizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class WarpSizeOpGenericAdaptor : public detail::WarpSizeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WarpSizeOpGenericAdaptorBase;
public:
  WarpSizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WarpSizeOpAdaptor : public WarpSizeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WarpSizeOpGenericAdaptor::WarpSizeOpGenericAdaptor;
  WarpSizeOpAdaptor(WarpSizeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WarpSizeOp : public ::mlir::Op<WarpSizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WarpSizeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WarpSizeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.warpsize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WarpSizeOp)


#endif  // GET_OP_CLASSES

