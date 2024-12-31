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
namespace amdgpu {
class LDSBarrierOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class MFMAOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferAtomicFaddOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferAtomicFmaxOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferAtomicSmaxOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferAtomicUminOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferLoadOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferStoreOp;
} // namespace amdgpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::LDSBarrierOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class LDSBarrierOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  LDSBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class LDSBarrierOpGenericAdaptor : public detail::LDSBarrierOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LDSBarrierOpGenericAdaptorBase;
public:
  LDSBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class LDSBarrierOpAdaptor : public LDSBarrierOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LDSBarrierOpGenericAdaptor::LDSBarrierOpGenericAdaptor;
  LDSBarrierOpAdaptor(LDSBarrierOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class LDSBarrierOp : public ::mlir::Op<LDSBarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LDSBarrierOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LDSBarrierOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.lds_barrier");
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::LDSBarrierOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::MFMAOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MFMAOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MFMAOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::IntegerAttr getBlocksAttr();
  uint32_t getBlocks();
  ::mlir::IntegerAttr getCbszAttr();
  uint32_t getCbsz();
  ::mlir::IntegerAttr getAbidAttr();
  uint32_t getAbid();
  ::mlir::amdgpu::MFMAPermBAttr getBlgpAttr();
  ::mlir::amdgpu::MFMAPermB getBlgp();
  ::mlir::UnitAttr getReducePrecisionAttr();
  bool getReducePrecision();
  ::mlir::UnitAttr getNegateAAttr();
  bool getNegateA();
  ::mlir::UnitAttr getNegateBAttr();
  bool getNegateB();
  ::mlir::UnitAttr getNegateCAttr();
  bool getNegateC();
};
} // namespace detail
template <typename RangeT>
class MFMAOpGenericAdaptor : public detail::MFMAOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MFMAOpGenericAdaptorBase;
public:
  MFMAOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSourceA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getSourceB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDestC() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MFMAOpAdaptor : public MFMAOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MFMAOpGenericAdaptor::MFMAOpGenericAdaptor;
  MFMAOpAdaptor(MFMAOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MFMAOp : public ::mlir::Op<MFMAOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MFMAOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MFMAOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("abid"), ::llvm::StringRef("blgp"), ::llvm::StringRef("blocks"), ::llvm::StringRef("cbsz"), ::llvm::StringRef("k"), ::llvm::StringRef("m"), ::llvm::StringRef("n"), ::llvm::StringRef("negateA"), ::llvm::StringRef("negateB"), ::llvm::StringRef("negateC"), ::llvm::StringRef("reducePrecision")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAbidAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAbidAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBlgpAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBlgpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBlocksAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBlocksAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCbszAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCbszAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getKAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getKAttrName(::mlir::OperationName name) {
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

  ::mlir::StringAttr getNegateAAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getNegateAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getNegateBAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getNegateBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getNegateCAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getNegateCAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getReducePrecisionAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getReducePrecisionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.mfma");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getSourceA();
  ::mlir::Value getSourceB();
  ::mlir::Value getDestC();
  ::mlir::MutableOperandRange getSourceAMutable();
  ::mlir::MutableOperandRange getSourceBMutable();
  ::mlir::MutableOperandRange getDestCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getDestD();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::IntegerAttr getBlocksAttr();
  uint32_t getBlocks();
  ::mlir::IntegerAttr getCbszAttr();
  uint32_t getCbsz();
  ::mlir::IntegerAttr getAbidAttr();
  uint32_t getAbid();
  ::mlir::amdgpu::MFMAPermBAttr getBlgpAttr();
  ::mlir::amdgpu::MFMAPermB getBlgp();
  ::mlir::UnitAttr getReducePrecisionAttr();
  bool getReducePrecision();
  ::mlir::UnitAttr getNegateAAttr();
  bool getNegateA();
  ::mlir::UnitAttr getNegateBAttr();
  bool getNegateB();
  ::mlir::UnitAttr getNegateCAttr();
  bool getNegateC();
  void setMAttr(::mlir::IntegerAttr attr);
  void setM(uint32_t attrValue);
  void setNAttr(::mlir::IntegerAttr attr);
  void setN(uint32_t attrValue);
  void setKAttr(::mlir::IntegerAttr attr);
  void setK(uint32_t attrValue);
  void setBlocksAttr(::mlir::IntegerAttr attr);
  void setBlocks(uint32_t attrValue);
  void setCbszAttr(::mlir::IntegerAttr attr);
  void setCbsz(uint32_t attrValue);
  void setAbidAttr(::mlir::IntegerAttr attr);
  void setAbid(uint32_t attrValue);
  void setBlgpAttr(::mlir::amdgpu::MFMAPermBAttr attr);
  void setBlgp(::mlir::amdgpu::MFMAPermB attrValue);
  void setReducePrecisionAttr(::mlir::UnitAttr attr);
  void setReducePrecision(bool attrValue);
  void setNegateAAttr(::mlir::UnitAttr attr);
  void setNegateA(bool attrValue);
  void setNegateBAttr(::mlir::UnitAttr attr);
  void setNegateB(bool attrValue);
  void setNegateCAttr(::mlir::UnitAttr attr);
  void setNegateC(bool attrValue);
  ::mlir::Attribute removeReducePrecisionAttr();
  ::mlir::Attribute removeNegateAAttr();
  ::mlir::Attribute removeNegateBAttr();
  ::mlir::Attribute removeNegateCAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type destD, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::IntegerAttr blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, ::mlir::IntegerAttr cbsz, ::mlir::IntegerAttr abid, ::mlir::amdgpu::MFMAPermBAttr blgp, /*optional*/::mlir::UnitAttr reducePrecision, /*optional*/::mlir::UnitAttr negateA, /*optional*/::mlir::UnitAttr negateB, /*optional*/::mlir::UnitAttr negateC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::IntegerAttr blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, ::mlir::IntegerAttr cbsz, ::mlir::IntegerAttr abid, ::mlir::amdgpu::MFMAPermBAttr blgp, /*optional*/::mlir::UnitAttr reducePrecision, /*optional*/::mlir::UnitAttr negateA, /*optional*/::mlir::UnitAttr negateB, /*optional*/::mlir::UnitAttr negateC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type destD, uint32_t m, uint32_t n, uint32_t k, uint32_t blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, uint32_t cbsz = 0, uint32_t abid = 0, ::mlir::amdgpu::MFMAPermB blgp = ::mlir::amdgpu::MFMAPermB::none, /*optional*/bool reducePrecision = false, /*optional*/bool negateA = false, /*optional*/bool negateB = false, /*optional*/bool negateC = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t m, uint32_t n, uint32_t k, uint32_t blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, uint32_t cbsz = 0, uint32_t abid = 0, ::mlir::amdgpu::MFMAPermB blgp = ::mlir::amdgpu::MFMAPermB::none, /*optional*/bool reducePrecision = false, /*optional*/bool negateA = false, /*optional*/bool negateB = false, /*optional*/bool negateC = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::MFMAOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicFaddOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RawBufferAtomicFaddOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RawBufferAtomicFaddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
};
} // namespace detail
template <typename RangeT>
class RawBufferAtomicFaddOpGenericAdaptor : public detail::RawBufferAtomicFaddOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RawBufferAtomicFaddOpGenericAdaptorBase;
public:
  RawBufferAtomicFaddOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  ValueT getSgprOffset() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RawBufferAtomicFaddOpAdaptor : public RawBufferAtomicFaddOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RawBufferAtomicFaddOpGenericAdaptor::RawBufferAtomicFaddOpGenericAdaptor;
  RawBufferAtomicFaddOpAdaptor(RawBufferAtomicFaddOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RawBufferAtomicFaddOp : public ::mlir::Op<RawBufferAtomicFaddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferAtomicFaddOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RawBufferAtomicFaddOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_atomic_fadd");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::FloatType> getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setBoundsCheck(bool attrValue);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  void setIndexOffset(::std::optional<uint32_t> attrValue);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicFaddOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicFmaxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RawBufferAtomicFmaxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RawBufferAtomicFmaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
};
} // namespace detail
template <typename RangeT>
class RawBufferAtomicFmaxOpGenericAdaptor : public detail::RawBufferAtomicFmaxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RawBufferAtomicFmaxOpGenericAdaptorBase;
public:
  RawBufferAtomicFmaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  ValueT getSgprOffset() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RawBufferAtomicFmaxOpAdaptor : public RawBufferAtomicFmaxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RawBufferAtomicFmaxOpGenericAdaptor::RawBufferAtomicFmaxOpGenericAdaptor;
  RawBufferAtomicFmaxOpAdaptor(RawBufferAtomicFmaxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RawBufferAtomicFmaxOp : public ::mlir::Op<RawBufferAtomicFmaxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferAtomicFmaxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RawBufferAtomicFmaxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_atomic_fmax");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::FloatType> getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setBoundsCheck(bool attrValue);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  void setIndexOffset(::std::optional<uint32_t> attrValue);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicFmaxOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicSmaxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RawBufferAtomicSmaxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RawBufferAtomicSmaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
};
} // namespace detail
template <typename RangeT>
class RawBufferAtomicSmaxOpGenericAdaptor : public detail::RawBufferAtomicSmaxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RawBufferAtomicSmaxOpGenericAdaptorBase;
public:
  RawBufferAtomicSmaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  ValueT getSgprOffset() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RawBufferAtomicSmaxOpAdaptor : public RawBufferAtomicSmaxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RawBufferAtomicSmaxOpGenericAdaptor::RawBufferAtomicSmaxOpGenericAdaptor;
  RawBufferAtomicSmaxOpAdaptor(RawBufferAtomicSmaxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RawBufferAtomicSmaxOp : public ::mlir::Op<RawBufferAtomicSmaxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferAtomicSmaxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RawBufferAtomicSmaxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_atomic_smax");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setBoundsCheck(bool attrValue);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  void setIndexOffset(::std::optional<uint32_t> attrValue);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicSmaxOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicUminOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RawBufferAtomicUminOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RawBufferAtomicUminOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
};
} // namespace detail
template <typename RangeT>
class RawBufferAtomicUminOpGenericAdaptor : public detail::RawBufferAtomicUminOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RawBufferAtomicUminOpGenericAdaptorBase;
public:
  RawBufferAtomicUminOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  ValueT getSgprOffset() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RawBufferAtomicUminOpAdaptor : public RawBufferAtomicUminOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RawBufferAtomicUminOpGenericAdaptor::RawBufferAtomicUminOpGenericAdaptor;
  RawBufferAtomicUminOpAdaptor(RawBufferAtomicUminOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RawBufferAtomicUminOp : public ::mlir::Op<RawBufferAtomicUminOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferAtomicUminOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RawBufferAtomicUminOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_atomic_umin");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setBoundsCheck(bool attrValue);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  void setIndexOffset(::std::optional<uint32_t> attrValue);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicUminOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RawBufferLoadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RawBufferLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
};
} // namespace detail
template <typename RangeT>
class RawBufferLoadOpGenericAdaptor : public detail::RawBufferLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RawBufferLoadOpGenericAdaptorBase;
public:
  RawBufferLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  ValueT getSgprOffset() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RawBufferLoadOpAdaptor : public RawBufferLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RawBufferLoadOpGenericAdaptor::RawBufferLoadOpGenericAdaptor;
  RawBufferLoadOpAdaptor(RawBufferLoadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RawBufferLoadOp : public ::mlir::Op<RawBufferLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RawBufferLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getValue();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setBoundsCheck(bool attrValue);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  void setIndexOffset(::std::optional<uint32_t> attrValue);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferLoadOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RawBufferStoreOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RawBufferStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
};
} // namespace detail
template <typename RangeT>
class RawBufferStoreOpGenericAdaptor : public detail::RawBufferStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RawBufferStoreOpGenericAdaptorBase;
public:
  RawBufferStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  ValueT getSgprOffset() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RawBufferStoreOpAdaptor : public RawBufferStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RawBufferStoreOpGenericAdaptor::RawBufferStoreOpGenericAdaptor;
  RawBufferStoreOpAdaptor(RawBufferStoreOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RawBufferStoreOp : public ::mlir::Op<RawBufferStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RawBufferStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::std::optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setBoundsCheck(bool attrValue);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  void setIndexOffset(::std::optional<uint32_t> attrValue);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
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
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferStoreOp)


#endif  // GET_OP_CLASSES

