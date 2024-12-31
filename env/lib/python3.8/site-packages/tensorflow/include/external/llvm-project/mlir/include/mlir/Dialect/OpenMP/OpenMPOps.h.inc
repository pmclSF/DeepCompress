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
namespace omp {
class AtomicCaptureOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicReadOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicUpdateOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicWriteOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class BarrierOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CancelOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CancellationPointOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class FlushOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class MasterOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OrderedOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OrderedRegionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ParallelOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SectionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SectionsOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SimdLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SingleOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TargetOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class DataOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class EnterDataOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ExitDataOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskGroupOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskwaitOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskyieldOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TerminatorOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ThreadprivateOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class WsLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class YieldOp;
} // namespace omp
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicCaptureOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AtomicCaptureOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AtomicCaptureOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AtomicCaptureOpGenericAdaptor : public detail::AtomicCaptureOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicCaptureOpGenericAdaptorBase;
public:
  AtomicCaptureOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class AtomicCaptureOpAdaptor : public AtomicCaptureOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicCaptureOpGenericAdaptor::AtomicCaptureOpGenericAdaptor;
  AtomicCaptureOpAdaptor(AtomicCaptureOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AtomicCaptureOp : public ::mlir::Op<AtomicCaptureOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<TerminatorOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicCaptureOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicCaptureOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.capture");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setHintVal(::std::optional<uint64_t> attrValue);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  void setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue);
  ::mlir::Attribute removeHintValAttr();
  ::mlir::Attribute removeMemoryOrderValAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
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
  /// Returns the first operation in atomic capture region
  Operation* getFirstOp();

  /// Returns the second operation in atomic capture region
  Operation* getSecondOp();

  /// Returns the `atomic.read` operation inside the region, if any.
  /// Otherwise, it returns nullptr.
  AtomicReadOp getAtomicReadOp();

  /// Returns the `atomic.write` operation inside the region, if any.
  /// Otherwise, it returns nullptr.
  AtomicWriteOp getAtomicWriteOp();

  /// Returns the `atomic.update` operation inside the region, if any.
  /// Otherwise, it returns nullptr.
  AtomicUpdateOp getAtomicUpdateOp();
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicCaptureOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicReadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AtomicReadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AtomicReadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::TypeAttr getElementTypeAttr();
  ::mlir::Type getElementType();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
};
} // namespace detail
template <typename RangeT>
class AtomicReadOpGenericAdaptor : public detail::AtomicReadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicReadOpGenericAdaptorBase;
public:
  AtomicReadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getX() {
    return (*getODSOperands(0).begin());
  }

  ValueT getV() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AtomicReadOpAdaptor : public AtomicReadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicReadOpGenericAdaptor::AtomicReadOpGenericAdaptor;
  AtomicReadOpAdaptor(AtomicReadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AtomicReadOp : public ::mlir::Op<AtomicReadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicReadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicReadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("element_type"), ::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getElementTypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getElementTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.read");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getX();
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getV();
  ::mlir::MutableOperandRange getXMutable();
  ::mlir::MutableOperandRange getVMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypeAttr getElementTypeAttr();
  ::mlir::Type getElementType();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setElementTypeAttr(::mlir::TypeAttr attr);
  void setElementType(::mlir::Type attrValue);
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setHintVal(::std::optional<uint64_t> attrValue);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  void setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue);
  ::mlir::Attribute removeHintValAttr();
  ::mlir::Attribute removeMemoryOrderValAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, ::mlir::TypeAttr element_type, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::TypeAttr element_type, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, ::mlir::Type element_type, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::Type element_type, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getX() && "expected 'x' operand");
    assert(getV() && "expected 'v' operand");
    return 2;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i < 2 && "invalid index position for an operand");
    return i == 0 ? getX() : getV();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicReadOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicUpdateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AtomicUpdateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AtomicUpdateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AtomicUpdateOpGenericAdaptor : public detail::AtomicUpdateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicUpdateOpGenericAdaptorBase;
public:
  AtomicUpdateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getX() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AtomicUpdateOpAdaptor : public AtomicUpdateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicUpdateOpGenericAdaptor::AtomicUpdateOpGenericAdaptor;
  AtomicUpdateOpAdaptor(AtomicUpdateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AtomicUpdateOp : public ::mlir::Op<AtomicUpdateOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicUpdateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicUpdateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getX();
  ::mlir::MutableOperandRange getXMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setHintVal(::std::optional<uint64_t> attrValue);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  void setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue);
  ::mlir::Attribute removeHintValAttr();
  ::mlir::Attribute removeMemoryOrderValAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static ::mlir::LogicalResult canonicalize(AtomicUpdateOp op, ::mlir::PatternRewriter &rewriter);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
  Operation* getFirstOp() {
    return &getRegion().front().getOperations().front();
  }

  /// Returns true if the new value is same as old value and the operation is
  /// a no-op, false otherwise.
  bool isNoOp();

  /// Returns the new value if the operation is equivalent to just a write
  /// operation. Otherwise, returns nullptr.
  Value getWriteOpVal();

  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getX() && "expected 'x' operand");
    return 1;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i == 0 && "invalid index position for an operand");
    return getX();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicUpdateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicWriteOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AtomicWriteOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AtomicWriteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
};
} // namespace detail
template <typename RangeT>
class AtomicWriteOpGenericAdaptor : public detail::AtomicWriteOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicWriteOpGenericAdaptorBase;
public:
  AtomicWriteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAddress() {
    return (*getODSOperands(0).begin());
  }

  ValueT getValue() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AtomicWriteOpAdaptor : public AtomicWriteOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicWriteOpGenericAdaptor::AtomicWriteOpGenericAdaptor;
  AtomicWriteOpAdaptor(AtomicWriteOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AtomicWriteOp : public ::mlir::Op<AtomicWriteOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicWriteOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicWriteOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.write");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getAddress();
  ::mlir::Value getValue();
  ::mlir::MutableOperandRange getAddressMutable();
  ::mlir::MutableOperandRange getValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::std::optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setHintVal(::std::optional<uint64_t> attrValue);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  void setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue);
  ::mlir::Attribute removeHintValAttr();
  ::mlir::Attribute removeMemoryOrderValAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getAddress() && "expected address operand");
    assert(getValue() && "expected value operand");
    return 2;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i < 2 && "invalid index position for an operand");
    return i == 0 ? getAddress() : getValue();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicWriteOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BarrierOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BarrierOpGenericAdaptor : public detail::BarrierOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BarrierOpGenericAdaptorBase;
public:
  BarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class BarrierOpAdaptor : public BarrierOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BarrierOpGenericAdaptor::BarrierOpGenericAdaptor;
  BarrierOpAdaptor(BarrierOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BarrierOp : public ::mlir::Op<BarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarrierOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BarrierOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.barrier");
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CancelOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CancelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
};
} // namespace detail
template <typename RangeT>
class CancelOpGenericAdaptor : public detail::CancelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CancelOpGenericAdaptorBase;
public:
  CancelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CancelOpAdaptor : public CancelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CancelOpGenericAdaptor::CancelOpGenericAdaptor;
  CancelOpAdaptor(CancelOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CancelOp : public ::mlir::Op<CancelOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CancelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CancelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cancellation_construct_type_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCancellationConstructTypeValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCancellationConstructTypeValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.cancel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::MutableOperandRange getIfExprMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
  void setCancellationConstructTypeValAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr);
  void setCancellationConstructTypeVal(::mlir::omp::ClauseCancellationConstructType attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
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
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CancelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancellationPointOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CancellationPointOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CancellationPointOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
};
} // namespace detail
template <typename RangeT>
class CancellationPointOpGenericAdaptor : public detail::CancellationPointOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CancellationPointOpGenericAdaptorBase;
public:
  CancellationPointOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class CancellationPointOpAdaptor : public CancellationPointOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CancellationPointOpGenericAdaptor::CancellationPointOpGenericAdaptor;
  CancellationPointOpAdaptor(CancellationPointOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CancellationPointOp : public ::mlir::Op<CancellationPointOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CancellationPointOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CancellationPointOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cancellation_construct_type_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCancellationConstructTypeValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCancellationConstructTypeValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.cancellationpoint");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
  void setCancellationConstructTypeValAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr);
  void setCancellationConstructTypeVal(::mlir::omp::ClauseCancellationConstructType attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val);
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
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CancellationPointOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CriticalDeclareOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CriticalDeclareOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
};
} // namespace detail
template <typename RangeT>
class CriticalDeclareOpGenericAdaptor : public detail::CriticalDeclareOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CriticalDeclareOpGenericAdaptorBase;
public:
  CriticalDeclareOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class CriticalDeclareOpAdaptor : public CriticalDeclareOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CriticalDeclareOpGenericAdaptor::CriticalDeclareOpGenericAdaptor;
  CriticalDeclareOpAdaptor(CriticalDeclareOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CriticalDeclareOp : public ::mlir::Op<CriticalDeclareOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalDeclareOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CriticalDeclareOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("sym_name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical.declare");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setSymName(::llvm::StringRef attrValue);
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setHintVal(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint64_t hint_val = 0);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t hint_val = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CriticalOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CriticalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class CriticalOpGenericAdaptor : public detail::CriticalOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CriticalOpGenericAdaptorBase;
public:
  CriticalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class CriticalOpAdaptor : public CriticalOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CriticalOpGenericAdaptor::CriticalOpGenericAdaptor;
  CriticalOpAdaptor(CriticalOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CriticalOp : public ::mlir::Op<CriticalOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CriticalOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::FlatSymbolRefAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setNameAttr(::mlir::FlatSymbolRefAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class FlushOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  FlushOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class FlushOpGenericAdaptor : public detail::FlushOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FlushOpGenericAdaptorBase;
public:
  FlushOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getVarList() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FlushOpAdaptor : public FlushOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FlushOpGenericAdaptor::FlushOpGenericAdaptor;
  FlushOpAdaptor(FlushOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class FlushOp : public ::mlir::Op<FlushOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FlushOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FlushOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.flush");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getVarList();
  ::mlir::MutableOperandRange getVarListMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    return getOperation()->getNumOperands();
  }
  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    return getOperand(i);
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MasterOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MasterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class MasterOpGenericAdaptor : public detail::MasterOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MasterOpGenericAdaptorBase;
public:
  MasterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class MasterOpAdaptor : public MasterOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MasterOpGenericAdaptor::MasterOpGenericAdaptor;
  MasterOpAdaptor(MasterOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MasterOp : public ::mlir::Op<MasterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MasterOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MasterOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.master");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class OrderedOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  OrderedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseDependAttr getDependTypeValAttr();
  ::std::optional<::mlir::omp::ClauseDepend> getDependTypeVal();
  ::mlir::IntegerAttr getNumLoopsValAttr();
  ::std::optional<uint64_t> getNumLoopsVal();
};
} // namespace detail
template <typename RangeT>
class OrderedOpGenericAdaptor : public detail::OrderedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::OrderedOpGenericAdaptorBase;
public:
  OrderedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getDependVecVars() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class OrderedOpAdaptor : public OrderedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using OrderedOpGenericAdaptor::OrderedOpGenericAdaptor;
  OrderedOpAdaptor(OrderedOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class OrderedOp : public ::mlir::Op<OrderedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OrderedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = OrderedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("depend_type_val"), ::llvm::StringRef("num_loops_val")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDependTypeValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDependTypeValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumLoopsValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumLoopsValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.ordered");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDependVecVars();
  ::mlir::MutableOperandRange getDependVecVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseDependAttr getDependTypeValAttr();
  ::std::optional<::mlir::omp::ClauseDepend> getDependTypeVal();
  ::mlir::IntegerAttr getNumLoopsValAttr();
  ::std::optional<uint64_t> getNumLoopsVal();
  void setDependTypeValAttr(::mlir::omp::ClauseDependAttr attr);
  void setDependTypeVal(::std::optional<::mlir::omp::ClauseDepend> attrValue);
  void setNumLoopsValAttr(::mlir::IntegerAttr attr);
  void setNumLoopsVal(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeDependTypeValAttr();
  ::mlir::Attribute removeNumLoopsValAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedRegionOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class OrderedRegionOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  OrderedRegionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getSimdAttr();
  bool getSimd();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class OrderedRegionOpGenericAdaptor : public detail::OrderedRegionOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::OrderedRegionOpGenericAdaptorBase;
public:
  OrderedRegionOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class OrderedRegionOpAdaptor : public OrderedRegionOpGenericAdaptor<::mlir::ValueRange> {
public:
  using OrderedRegionOpGenericAdaptor::OrderedRegionOpGenericAdaptor;
  OrderedRegionOpAdaptor(OrderedRegionOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class OrderedRegionOp : public ::mlir::Op<OrderedRegionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OrderedRegionOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = OrderedRegionOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("simd")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getSimdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSimdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.ordered_region");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getSimdAttr();
  bool getSimd();
  void setSimdAttr(::mlir::UnitAttr attr);
  void setSimd(bool attrValue);
  ::mlir::Attribute removeSimdAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool simd = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool simd = false);
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
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedRegionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ParallelOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseProcBindKindAttr getProcBindValAttr();
  ::std::optional<::mlir::omp::ClauseProcBindKind> getProcBindVal();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ParallelOpGenericAdaptor : public detail::ParallelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ParallelOpGenericAdaptorBase;
public:
  ParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExprVar() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getNumThreadsVar() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getAllocateVars() {
    return getODSOperands(2);
  }

  RangeT getAllocatorsVars() {
    return getODSOperands(3);
  }

  RangeT getReductionVars() {
    return getODSOperands(4);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ParallelOpAdaptor : public ParallelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ParallelOpGenericAdaptor::ParallelOpGenericAdaptor;
  ParallelOpAdaptor(ParallelOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::omp::OutlineableOpenMPOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("proc_bind_val"), ::llvm::StringRef("reductions")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getProcBindValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getProcBindValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExprVar();
  ::mlir::Value getNumThreadsVar();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::MutableOperandRange getIfExprVarMutable();
  ::mlir::MutableOperandRange getNumThreadsVarMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseProcBindKindAttr getProcBindValAttr();
  ::std::optional<::mlir::omp::ClauseProcBindKind> getProcBindVal();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setProcBindValAttr(::mlir::omp::ClauseProcBindKindAttr attr);
  void setProcBindVal(::std::optional<::mlir::omp::ClauseProcBindKind> attrValue);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeProcBindValAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val);
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReductionDeclareOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReductionDeclareOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getTypeAttr();
  ::mlir::Type getType();
  ::mlir::Region &getInitializerRegion();
  ::mlir::Region &getReductionRegion();
  ::mlir::Region &getAtomicReductionRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ReductionDeclareOpGenericAdaptor : public detail::ReductionDeclareOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReductionDeclareOpGenericAdaptorBase;
public:
  ReductionDeclareOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class ReductionDeclareOpAdaptor : public ReductionDeclareOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReductionDeclareOpGenericAdaptor::ReductionDeclareOpGenericAdaptor;
  ReductionDeclareOpAdaptor(ReductionDeclareOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReductionDeclareOp : public ::mlir::Op<ReductionDeclareOp, ::mlir::OpTrait::NRegions<3>::Impl, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionDeclareOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReductionDeclareOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name"), ::llvm::StringRef("type")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction.declare");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getInitializerRegion();
  ::mlir::Region &getReductionRegion();
  ::mlir::Region &getAtomicReductionRegion();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getTypeAttr();
  ::mlir::Type getType();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setSymName(::llvm::StringRef attrValue);
  void setTypeAttr(::mlir::TypeAttr attr);
  void setType(::mlir::Type attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifyRegions();
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
  PointerLikeType getAccumulatorType() {
    if (getAtomicReductionRegion().empty())
      return {};

    return cast<PointerLikeType>(getAtomicReductionRegion().front().getArgument(0).getType());
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReductionOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReductionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReductionOpGenericAdaptor : public detail::ReductionOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReductionOpGenericAdaptorBase;
public:
  ReductionOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getOperand() {
    return (*getODSOperands(0).begin());
  }

  ValueT getAccumulator() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReductionOpAdaptor : public ReductionOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReductionOpGenericAdaptor::ReductionOpGenericAdaptor;
  ReductionOpAdaptor(ReductionOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReductionOp : public ::mlir::Op<ReductionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReductionOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getAccumulator();
  ::mlir::MutableOperandRange getOperandMutable();
  ::mlir::MutableOperandRange getAccumulatorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SectionOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SectionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class SectionOpGenericAdaptor : public detail::SectionOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SectionOpGenericAdaptorBase;
public:
  SectionOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class SectionOpAdaptor : public SectionOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SectionOpGenericAdaptor::SectionOpGenericAdaptor;
  SectionOpAdaptor(SectionOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SectionOp : public ::mlir::Op<SectionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<SectionsOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SectionOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SectionOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.section");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SectionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SectionsOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SectionsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class SectionsOpGenericAdaptor : public detail::SectionsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SectionsOpGenericAdaptorBase;
public:
  SectionsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getReductionVars() {
    return getODSOperands(0);
  }

  RangeT getAllocateVars() {
    return getODSOperands(1);
  }

  RangeT getAllocatorsVars() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SectionsOpAdaptor : public SectionsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SectionsOpGenericAdaptor::SectionsOpGenericAdaptor;
  SectionsOpAdaptor(SectionsOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SectionsOp : public ::mlir::Op<SectionsOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SectionsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SectionsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductions")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.sections");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setNowait(bool attrValue);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SectionsOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SimdLoopOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SimdLoopOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SimdLoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getAlignmentValuesAttr();
  ::std::optional< ::mlir::ArrayAttr > getAlignmentValues();
  ::mlir::omp::ClauseOrderKindAttr getOrderValAttr();
  ::std::optional<::mlir::omp::ClauseOrderKind> getOrderVal();
  ::mlir::IntegerAttr getSimdlenAttr();
  ::std::optional<uint64_t> getSimdlen();
  ::mlir::IntegerAttr getSafelenAttr();
  ::std::optional<uint64_t> getSafelen();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class SimdLoopOpGenericAdaptor : public detail::SimdLoopOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SimdLoopOpGenericAdaptorBase;
public:
  SimdLoopOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getLowerBound() {
    return getODSOperands(0);
  }

  RangeT getUpperBound() {
    return getODSOperands(1);
  }

  RangeT getStep() {
    return getODSOperands(2);
  }

  RangeT getAlignedVars() {
    return getODSOperands(3);
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(4);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getNontemporalVars() {
    return getODSOperands(5);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SimdLoopOpAdaptor : public SimdLoopOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SimdLoopOpGenericAdaptor::SimdLoopOpGenericAdaptor;
  SimdLoopOpAdaptor(SimdLoopOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SimdLoopOp : public ::mlir::Op<SimdLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SimdLoopOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SimdLoopOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment_values"), ::llvm::StringRef("inclusive"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("order_val"), ::llvm::StringRef("safelen"), ::llvm::StringRef("simdlen")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentValuesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentValuesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getInclusiveAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getInclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOrderValAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSafelenAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSafelenAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getSimdlenAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getSimdlenAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.simdloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::Operation::operand_range getAlignedVars();
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::Operation::operand_range getNontemporalVars();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getAlignedVarsMutable();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getNontemporalVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getAlignmentValuesAttr();
  ::std::optional< ::mlir::ArrayAttr > getAlignmentValues();
  ::mlir::omp::ClauseOrderKindAttr getOrderValAttr();
  ::std::optional<::mlir::omp::ClauseOrderKind> getOrderVal();
  ::mlir::IntegerAttr getSimdlenAttr();
  ::std::optional<uint64_t> getSimdlen();
  ::mlir::IntegerAttr getSafelenAttr();
  ::std::optional<uint64_t> getSafelen();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  void setAlignmentValuesAttr(::mlir::ArrayAttr attr);
  void setOrderValAttr(::mlir::omp::ClauseOrderKindAttr attr);
  void setOrderVal(::std::optional<::mlir::omp::ClauseOrderKind> attrValue);
  void setSimdlenAttr(::mlir::IntegerAttr attr);
  void setSimdlen(::std::optional<uint64_t> attrValue);
  void setSafelenAttr(::mlir::IntegerAttr attr);
  void setSafelen(::std::optional<uint64_t> attrValue);
  void setInclusiveAttr(::mlir::UnitAttr attr);
  void setInclusive(bool attrValue);
  ::mlir::Attribute removeAlignmentValuesAttr();
  ::mlir::Attribute removeOrderValAttr();
  ::mlir::Attribute removeSimdlenAttr();
  ::mlir::Attribute removeSafelenAttr();
  ::mlir::Attribute removeInclusiveAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/bool inclusive = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/bool inclusive = false);
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
  /// Returns the number of loops in the simd loop nest.
  unsigned getNumLoops() { return getLowerBound().size(); }

};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SimdLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SingleOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SingleOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SingleOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class SingleOpGenericAdaptor : public detail::SingleOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SingleOpGenericAdaptorBase;
public:
  SingleOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getAllocateVars() {
    return getODSOperands(0);
  }

  RangeT getAllocatorsVars() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SingleOpAdaptor : public SingleOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SingleOpGenericAdaptor::SingleOpGenericAdaptor;
  SingleOpAdaptor(SingleOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SingleOp : public ::mlir::Op<SingleOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SingleOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SingleOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.single");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setNowait(bool attrValue);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SingleOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TargetOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TargetOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::std::optional< ::mlir::ArrayAttr > getMapTypes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class TargetOpGenericAdaptor : public detail::TargetOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TargetOpGenericAdaptorBase;
public:
  TargetOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getDevice() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getThreadLimit() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getMapOperands() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TargetOpAdaptor : public TargetOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TargetOpGenericAdaptor::TargetOpGenericAdaptor;
  TargetOpAdaptor(TargetOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TargetOp : public ::mlir::Op<TargetOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TargetOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TargetOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map_types"), ::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapTypesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapTypesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getDevice();
  ::mlir::TypedValue<::mlir::IntegerType> getThreadLimit();
  ::mlir::Operation::operand_range getMapOperands();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getDeviceMutable();
  ::mlir::MutableOperandRange getThreadLimitMutable();
  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::std::optional< ::mlir::ArrayAttr > getMapTypes();
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setNowait(bool attrValue);
  void setMapTypesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  ::mlir::Attribute removeMapTypesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types);
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::DataOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DataOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::mlir::ArrayAttr getMapTypes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class DataOpGenericAdaptor : public detail::DataOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DataOpGenericAdaptorBase;
public:
  DataOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getDevice() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getUseDevicePtr() {
    return getODSOperands(2);
  }

  RangeT getUseDeviceAddr() {
    return getODSOperands(3);
  }

  RangeT getMapOperands() {
    return getODSOperands(4);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DataOpAdaptor : public DataOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DataOpGenericAdaptor::DataOpGenericAdaptor;
  DataOpAdaptor(DataOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DataOp : public ::mlir::Op<DataOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DataOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DataOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map_types"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapTypesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapTypesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getDevice();
  ::mlir::Operation::operand_range getUseDevicePtr();
  ::mlir::Operation::operand_range getUseDeviceAddr();
  ::mlir::Operation::operand_range getMapOperands();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getDeviceMutable();
  ::mlir::MutableOperandRange getUseDevicePtrMutable();
  ::mlir::MutableOperandRange getUseDeviceAddrMutable();
  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::mlir::ArrayAttr getMapTypes();
  void setMapTypesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, ::mlir::ValueRange use_device_ptr, ::mlir::ValueRange use_device_addr, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, ::mlir::ValueRange use_device_ptr, ::mlir::ValueRange use_device_addr, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::DataOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::EnterDataOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class EnterDataOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  EnterDataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::mlir::ArrayAttr getMapTypes();
};
} // namespace detail
template <typename RangeT>
class EnterDataOpGenericAdaptor : public detail::EnterDataOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::EnterDataOpGenericAdaptorBase;
public:
  EnterDataOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getDevice() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getMapOperands() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class EnterDataOpAdaptor : public EnterDataOpGenericAdaptor<::mlir::ValueRange> {
public:
  using EnterDataOpGenericAdaptor::EnterDataOpGenericAdaptor;
  EnterDataOpAdaptor(EnterDataOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class EnterDataOp : public ::mlir::Op<EnterDataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EnterDataOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = EnterDataOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map_types"), ::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapTypesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapTypesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target_enter_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getDevice();
  ::mlir::Operation::operand_range getMapOperands();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getDeviceMutable();
  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::mlir::ArrayAttr getMapTypes();
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setNowait(bool attrValue);
  void setMapTypesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::EnterDataOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ExitDataOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExitDataOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExitDataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::mlir::ArrayAttr getMapTypes();
};
} // namespace detail
template <typename RangeT>
class ExitDataOpGenericAdaptor : public detail::ExitDataOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExitDataOpGenericAdaptorBase;
public:
  ExitDataOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getDevice() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getMapOperands() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExitDataOpAdaptor : public ExitDataOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExitDataOpGenericAdaptor::ExitDataOpGenericAdaptor;
  ExitDataOpAdaptor(ExitDataOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExitDataOp : public ::mlir::Op<ExitDataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExitDataOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExitDataOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map_types"), ::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapTypesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapTypesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target_exit_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getDevice();
  ::mlir::Operation::operand_range getMapOperands();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getDeviceMutable();
  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::ArrayAttr getMapTypesAttr();
  ::mlir::ArrayAttr getMapTypes();
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setNowait(bool attrValue);
  void setMapTypesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types);
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ExitDataOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskGroupOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TaskGroupOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TaskGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getTaskReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getTaskReductions();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class TaskGroupOpGenericAdaptor : public detail::TaskGroupOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TaskGroupOpGenericAdaptorBase;
public:
  TaskGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getTaskReductionVars() {
    return getODSOperands(0);
  }

  RangeT getAllocateVars() {
    return getODSOperands(1);
  }

  RangeT getAllocatorsVars() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TaskGroupOpAdaptor : public TaskGroupOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TaskGroupOpGenericAdaptor::TaskGroupOpGenericAdaptor;
  TaskGroupOpAdaptor(TaskGroupOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TaskGroupOp : public ::mlir::Op<TaskGroupOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::omp::ReductionClauseInterface::Trait, ::mlir::OpTrait::AutomaticAllocationScope> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskGroupOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TaskGroupOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("task_reductions")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTaskReductionsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTaskReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskgroup");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getTaskReductionVars();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getTaskReductionVarsMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getTaskReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getTaskReductions();
  void setTaskReductionsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTaskReductionsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange task_reduction_vars, /*optional*/::mlir::ArrayAttr task_reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange task_reduction_vars, /*optional*/::mlir::ArrayAttr task_reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
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
  /// Returns the reduction variables
  operand_range getAllReductionVars() { return getTaskReductionVars(); }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskGroupOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskLoopOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TaskLoopOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TaskLoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNogroupAttr();
  bool getNogroup();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class TaskLoopOpGenericAdaptor : public detail::TaskLoopOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TaskLoopOpGenericAdaptorBase;
public:
  TaskLoopOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getLowerBound() {
    return getODSOperands(0);
  }

  RangeT getUpperBound() {
    return getODSOperands(1);
  }

  RangeT getStep() {
    return getODSOperands(2);
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getFinalExpr() {
    auto operands = getODSOperands(4);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getInReductionVars() {
    return getODSOperands(5);
  }

  RangeT getReductionVars() {
    return getODSOperands(6);
  }

  ValueT getPriority() {
    auto operands = getODSOperands(7);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getAllocateVars() {
    return getODSOperands(8);
  }

  RangeT getAllocatorsVars() {
    return getODSOperands(9);
  }

  ValueT getGrainSize() {
    auto operands = getODSOperands(10);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getNumTasks() {
    auto operands = getODSOperands(11);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TaskLoopOpAdaptor : public TaskLoopOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TaskLoopOpGenericAdaptor::TaskLoopOpGenericAdaptor;
  TaskLoopOpAdaptor(TaskLoopOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TaskLoopOp : public ::mlir::Op<TaskLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskLoopOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TaskLoopOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("in_reductions"), ::llvm::StringRef("inclusive"), ::llvm::StringRef("mergeable"), ::llvm::StringRef("nogroup"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductions"), ::llvm::StringRef("untied")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInReductionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getInclusiveAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getInclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMergeableAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMergeableAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getNogroupAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getNogroupAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getUntiedAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getUntiedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getFinalExpr();
  ::mlir::Operation::operand_range getInReductionVars();
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::Value getPriority();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::Value getGrainSize();
  ::mlir::Value getNumTasks();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getFinalExprMutable();
  ::mlir::MutableOperandRange getInReductionVarsMutable();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  ::mlir::MutableOperandRange getPriorityMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  ::mlir::MutableOperandRange getGrainSizeMutable();
  ::mlir::MutableOperandRange getNumTasksMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNogroupAttr();
  bool getNogroup();
  void setInclusiveAttr(::mlir::UnitAttr attr);
  void setInclusive(bool attrValue);
  void setUntiedAttr(::mlir::UnitAttr attr);
  void setUntied(bool attrValue);
  void setMergeableAttr(::mlir::UnitAttr attr);
  void setMergeable(bool attrValue);
  void setInReductionsAttr(::mlir::ArrayAttr attr);
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setNogroupAttr(::mlir::UnitAttr attr);
  void setNogroup(bool attrValue);
  ::mlir::Attribute removeInclusiveAttr();
  ::mlir::Attribute removeUntiedAttr();
  ::mlir::Attribute removeMergeableAttr();
  ::mlir::Attribute removeInReductionsAttr();
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeNogroupAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::UnitAttr inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/::mlir::UnitAttr nogroup);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::UnitAttr inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/::mlir::UnitAttr nogroup);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/bool inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/bool nogroup = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/bool inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/bool nogroup = false);
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
  /// Returns the reduction variables
  SmallVector<Value> getAllReductionVars();
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TaskOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TaskOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::ArrayAttr getDependsAttr();
  ::std::optional< ::mlir::ArrayAttr > getDepends();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class TaskOpGenericAdaptor : public detail::TaskOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TaskOpGenericAdaptorBase;
public:
  TaskOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfExpr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getFinalExpr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getInReductionVars() {
    return getODSOperands(2);
  }

  ValueT getPriority() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getDependVars() {
    return getODSOperands(4);
  }

  RangeT getAllocateVars() {
    return getODSOperands(5);
  }

  RangeT getAllocatorsVars() {
    return getODSOperands(6);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TaskOpAdaptor : public TaskOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TaskOpGenericAdaptor::TaskOpGenericAdaptor;
  TaskOpAdaptor(TaskOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TaskOp : public ::mlir::Op<TaskOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::omp::OutlineableOpenMPOpInterface::Trait, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TaskOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("depends"), ::llvm::StringRef("in_reductions"), ::llvm::StringRef("mergeable"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("untied")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDependsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDependsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getInReductionsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getInReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMergeableAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMergeableAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getUntiedAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getUntiedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.task");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getFinalExpr();
  ::mlir::Operation::operand_range getInReductionVars();
  ::mlir::TypedValue<::mlir::IntegerType> getPriority();
  ::mlir::Operation::operand_range getDependVars();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getFinalExprMutable();
  ::mlir::MutableOperandRange getInReductionVarsMutable();
  ::mlir::MutableOperandRange getPriorityMutable();
  ::mlir::MutableOperandRange getDependVarsMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::ArrayAttr getDependsAttr();
  ::std::optional< ::mlir::ArrayAttr > getDepends();
  void setUntiedAttr(::mlir::UnitAttr attr);
  void setUntied(bool attrValue);
  void setMergeableAttr(::mlir::UnitAttr attr);
  void setMergeable(bool attrValue);
  void setInReductionsAttr(::mlir::ArrayAttr attr);
  void setDependsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeUntiedAttr();
  ::mlir::Attribute removeMergeableAttr();
  ::mlir::Attribute removeInReductionsAttr();
  ::mlir::Attribute removeDependsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
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
  /// Returns the reduction variables
  SmallVector<Value> getReductionVars() {
    return SmallVector<Value>(getInReductionVars().begin(),
                              getInReductionVars().end());
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TaskwaitOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TaskwaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class TaskwaitOpGenericAdaptor : public detail::TaskwaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TaskwaitOpGenericAdaptorBase;
public:
  TaskwaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class TaskwaitOpAdaptor : public TaskwaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TaskwaitOpGenericAdaptor::TaskwaitOpGenericAdaptor;
  TaskwaitOpAdaptor(TaskwaitOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TaskwaitOp : public ::mlir::Op<TaskwaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskwaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TaskwaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskwait");
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TaskyieldOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TaskyieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class TaskyieldOpGenericAdaptor : public detail::TaskyieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TaskyieldOpGenericAdaptorBase;
public:
  TaskyieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class TaskyieldOpAdaptor : public TaskyieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TaskyieldOpGenericAdaptor::TaskyieldOpGenericAdaptor;
  TaskyieldOpAdaptor(TaskyieldOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TaskyieldOp : public ::mlir::Op<TaskyieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskyieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TaskyieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskyield");
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TerminatorOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TerminatorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class TerminatorOpGenericAdaptor : public detail::TerminatorOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TerminatorOpGenericAdaptorBase;
public:
  TerminatorOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class TerminatorOpAdaptor : public TerminatorOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TerminatorOpGenericAdaptor::TerminatorOpGenericAdaptor;
  TerminatorOpAdaptor(TerminatorOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TerminatorOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.terminator");
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
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ThreadprivateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ThreadprivateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ThreadprivateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ThreadprivateOpGenericAdaptor : public detail::ThreadprivateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ThreadprivateOpGenericAdaptorBase;
public:
  ThreadprivateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSymAddr() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ThreadprivateOpAdaptor : public ThreadprivateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ThreadprivateOpGenericAdaptor::ThreadprivateOpGenericAdaptor;
  ThreadprivateOpAdaptor(ThreadprivateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ThreadprivateOp : public ::mlir::Op<ThreadprivateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::omp::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadprivateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ThreadprivateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.threadprivate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getSymAddr();
  ::mlir::MutableOperandRange getSymAddrMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getTlsAddr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tls_addr, ::mlir::Value sym_addr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sym_addr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getSymAddr() && "expected one variable operand");
    return 1;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i == 0 && "invalid index position for an operand");
    return getSymAddr();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ThreadprivateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WsLoopOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WsLoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseScheduleKindAttr getScheduleValAttr();
  ::std::optional<::mlir::omp::ClauseScheduleKind> getScheduleVal();
  ::mlir::omp::ScheduleModifierAttr getScheduleModifierAttr();
  ::std::optional<::mlir::omp::ScheduleModifier> getScheduleModifier();
  ::mlir::UnitAttr getSimdModifierAttr();
  bool getSimdModifier();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::IntegerAttr getOrderedValAttr();
  ::std::optional<uint64_t> getOrderedVal();
  ::mlir::omp::ClauseOrderKindAttr getOrderValAttr();
  ::std::optional<::mlir::omp::ClauseOrderKind> getOrderVal();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class WsLoopOpGenericAdaptor : public detail::WsLoopOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WsLoopOpGenericAdaptorBase;
public:
  WsLoopOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getLowerBound() {
    return getODSOperands(0);
  }

  RangeT getUpperBound() {
    return getODSOperands(1);
  }

  RangeT getStep() {
    return getODSOperands(2);
  }

  RangeT getLinearVars() {
    return getODSOperands(3);
  }

  RangeT getLinearStepVars() {
    return getODSOperands(4);
  }

  RangeT getReductionVars() {
    return getODSOperands(5);
  }

  ValueT getScheduleChunkVar() {
    auto operands = getODSOperands(6);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WsLoopOpAdaptor : public WsLoopOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WsLoopOpGenericAdaptor::WsLoopOpGenericAdaptor;
  WsLoopOpAdaptor(WsLoopOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WsLoopOp : public ::mlir::Op<WsLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WsLoopOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WsLoopOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inclusive"), ::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("order_val"), ::llvm::StringRef("ordered_val"), ::llvm::StringRef("reductions"), ::llvm::StringRef("schedule_modifier"), ::llvm::StringRef("schedule_val"), ::llvm::StringRef("simd_modifier")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInclusiveAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOrderValAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOrderedValAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOrderedValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getScheduleModifierAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getScheduleModifierAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getScheduleValAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getScheduleValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getSimdModifierAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getSimdModifierAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.wsloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::Operation::operand_range getLinearVars();
  ::mlir::Operation::operand_range getLinearStepVars();
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::Value getScheduleChunkVar();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getLinearVarsMutable();
  ::mlir::MutableOperandRange getLinearStepVarsMutable();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  ::mlir::MutableOperandRange getScheduleChunkVarMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::std::optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseScheduleKindAttr getScheduleValAttr();
  ::std::optional<::mlir::omp::ClauseScheduleKind> getScheduleVal();
  ::mlir::omp::ScheduleModifierAttr getScheduleModifierAttr();
  ::std::optional<::mlir::omp::ScheduleModifier> getScheduleModifier();
  ::mlir::UnitAttr getSimdModifierAttr();
  bool getSimdModifier();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::IntegerAttr getOrderedValAttr();
  ::std::optional<uint64_t> getOrderedVal();
  ::mlir::omp::ClauseOrderKindAttr getOrderValAttr();
  ::std::optional<::mlir::omp::ClauseOrderKind> getOrderVal();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setScheduleValAttr(::mlir::omp::ClauseScheduleKindAttr attr);
  void setScheduleVal(::std::optional<::mlir::omp::ClauseScheduleKind> attrValue);
  void setScheduleModifierAttr(::mlir::omp::ScheduleModifierAttr attr);
  void setScheduleModifier(::std::optional<::mlir::omp::ScheduleModifier> attrValue);
  void setSimdModifierAttr(::mlir::UnitAttr attr);
  void setSimdModifier(bool attrValue);
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setNowait(bool attrValue);
  void setOrderedValAttr(::mlir::IntegerAttr attr);
  void setOrderedVal(::std::optional<uint64_t> attrValue);
  void setOrderValAttr(::mlir::omp::ClauseOrderKindAttr attr);
  void setOrderVal(::std::optional<::mlir::omp::ClauseOrderKind> attrValue);
  void setInclusiveAttr(::mlir::UnitAttr attr);
  void setInclusive(bool attrValue);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeScheduleValAttr();
  ::mlir::Attribute removeScheduleModifierAttr();
  ::mlir::Attribute removeSimdModifierAttr();
  ::mlir::Attribute removeNowaitAttr();
  ::mlir::Attribute removeOrderedValAttr();
  ::mlir::Attribute removeOrderValAttr();
  ::mlir::Attribute removeInclusiveAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBound, ValueRange upperBound, ValueRange step, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive = false);
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
    assert(index < 9 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the number of loops in the worksharing-loop nest.
  unsigned getNumLoops() { return getLowerBound().size(); }

  /// Returns the number of reduction variables.
  unsigned getNumReductionVars() { return getReductionVars().size(); }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class YieldOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class YieldOpGenericAdaptor : public detail::YieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::YieldOpGenericAdaptorBase;
public:
  YieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getResults() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class YieldOpAdaptor : public YieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using YieldOpGenericAdaptor::YieldOpGenericAdaptor;
  YieldOpAdaptor(YieldOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<WsLoopOp, ReductionDeclareOp, AtomicUpdateOp, SimdLoopOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = YieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getResults();
  ::mlir::MutableOperandRange getResultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

