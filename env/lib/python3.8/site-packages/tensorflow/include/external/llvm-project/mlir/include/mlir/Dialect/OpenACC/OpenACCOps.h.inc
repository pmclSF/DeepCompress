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
namespace acc {
class AttachOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class CopyinOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class CopyoutOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class CreateOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class DataBoundsOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class DataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class DeleteOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class DetachOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class DevicePtrOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class EnterDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ExitDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class GetDevicePtrOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class InitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class KernelsOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class LoopOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class NoCreateOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ParallelOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class PresentOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class SerialOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ShutdownOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class TerminatorOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class UpdateOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class WaitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class YieldOp;
} // namespace acc
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::AttachOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AttachOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AttachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class AttachOpGenericAdaptor : public detail::AttachOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AttachOpGenericAdaptorBase;
public:
  AttachOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AttachOpAdaptor : public AttachOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AttachOpGenericAdaptor::AttachOpGenericAdaptor;
  AttachOpAdaptor(AttachOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AttachOp : public ::mlir::Op<AttachOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AttachOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AttachOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.attach");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::AttachOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::CopyinOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CopyinOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CopyinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class CopyinOpGenericAdaptor : public detail::CopyinOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CopyinOpGenericAdaptorBase;
public:
  CopyinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CopyinOpAdaptor : public CopyinOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CopyinOpGenericAdaptor::CopyinOpGenericAdaptor;
  CopyinOpAdaptor(CopyinOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CopyinOp : public ::mlir::Op<CopyinOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CopyinOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CopyinOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.copyin");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Check if this is a copyin with readonly modifier.
  bool isCopyinReadonly();
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::CopyinOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::CopyoutOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CopyoutOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CopyoutOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class CopyoutOpGenericAdaptor : public detail::CopyoutOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CopyoutOpGenericAdaptorBase;
public:
  CopyoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAccPtr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CopyoutOpAdaptor : public CopyoutOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CopyoutOpGenericAdaptor::CopyoutOpGenericAdaptor;
  CopyoutOpAdaptor(CopyoutOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CopyoutOp : public ::mlir::Op<CopyoutOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CopyoutOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CopyoutOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.copyout");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getAccPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Check if this is a copyout with zero modifier.
  bool isCopyoutZero();
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::CopyoutOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::CreateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CreateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CreateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class CreateOpGenericAdaptor : public detail::CreateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateOpGenericAdaptorBase;
public:
  CreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CreateOpAdaptor : public CreateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateOpGenericAdaptor::CreateOpGenericAdaptor;
  CreateOpAdaptor(CreateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CreateOp : public ::mlir::Op<CreateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.create");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Check if this is a create with zero modifier.
  bool isCreateZero();
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::CreateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataBoundsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DataBoundsOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DataBoundsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getStrideInBytesAttr();
  bool getStrideInBytes();
};
} // namespace detail
template <typename RangeT>
class DataBoundsOpGenericAdaptor : public detail::DataBoundsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DataBoundsOpGenericAdaptorBase;
public:
  DataBoundsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLowerbound() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getUpperbound() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getExtent() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getStride() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getStartIdx() {
    auto operands = getODSOperands(4);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DataBoundsOpAdaptor : public DataBoundsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DataBoundsOpGenericAdaptor::DataBoundsOpGenericAdaptor;
  DataBoundsOpAdaptor(DataBoundsOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DataBoundsOp : public ::mlir::Op<DataBoundsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::DataBoundsType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DataBoundsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DataBoundsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strideInBytes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStrideInBytesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStrideInBytesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.bounds");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getLowerbound();
  ::mlir::Value getUpperbound();
  ::mlir::Value getExtent();
  ::mlir::Value getStride();
  ::mlir::Value getStartIdx();
  ::mlir::MutableOperandRange getLowerboundMutable();
  ::mlir::MutableOperandRange getUpperboundMutable();
  ::mlir::MutableOperandRange getExtentMutable();
  ::mlir::MutableOperandRange getStrideMutable();
  ::mlir::MutableOperandRange getStartIdxMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::DataBoundsType> getResult();
  ::mlir::BoolAttr getStrideInBytesAttr();
  bool getStrideInBytes();
  void setStrideInBytesAttr(::mlir::BoolAttr attr);
  void setStrideInBytes(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, ::mlir::BoolAttr strideInBytes, /*optional*/::mlir::Value startIdx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, ::mlir::BoolAttr strideInBytes, /*optional*/::mlir::Value startIdx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, bool strideInBytes, /*optional*/::mlir::Value startIdx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, bool strideInBytes, /*optional*/::mlir::Value startIdx);
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
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataBoundsOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp declarations
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
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
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

  ValueT getIfCond() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getCopyOperands() {
    return getODSOperands(1);
  }

  RangeT getCopyinOperands() {
    return getODSOperands(2);
  }

  RangeT getCopyinReadonlyOperands() {
    return getODSOperands(3);
  }

  RangeT getCopyoutOperands() {
    return getODSOperands(4);
  }

  RangeT getCopyoutZeroOperands() {
    return getODSOperands(5);
  }

  RangeT getCreateOperands() {
    return getODSOperands(6);
  }

  RangeT getCreateZeroOperands() {
    return getODSOperands(7);
  }

  RangeT getNoCreateOperands() {
    return getODSOperands(8);
  }

  RangeT getPresentOperands() {
    return getODSOperands(9);
  }

  RangeT getDeviceptrOperands() {
    return getODSOperands(10);
  }

  RangeT getAttachOperands() {
    return getODSOperands(11);
  }

  RangeT getDataClauseOperands() {
    return getODSOperands(12);
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
class DataOp : public ::mlir::Op<DataOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DataOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DataOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDefaultAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDefaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Operation::operand_range getCopyOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCopyinReadonlyOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getCopyoutZeroOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getNoCreateOperands();
  ::mlir::Operation::operand_range getPresentOperands();
  ::mlir::Operation::operand_range getDeviceptrOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::Operation::operand_range getDataClauseOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getCopyOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCopyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getNoCreateOperandsMutable();
  ::mlir::MutableOperandRange getPresentOperandsMutable();
  ::mlir::MutableOperandRange getDeviceptrOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  ::mlir::MutableOperandRange getDataClauseOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  void setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  void setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue);
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
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
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DeleteOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DeleteOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DeleteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class DeleteOpGenericAdaptor : public detail::DeleteOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DeleteOpGenericAdaptorBase;
public:
  DeleteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAccPtr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DeleteOpAdaptor : public DeleteOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DeleteOpGenericAdaptor::DeleteOpGenericAdaptor;
  DeleteOpAdaptor(DeleteOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DeleteOp : public ::mlir::Op<DeleteOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeleteOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DeleteOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.delete");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getAccPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DeleteOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DetachOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DetachOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DetachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class DetachOpGenericAdaptor : public detail::DetachOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DetachOpGenericAdaptorBase;
public:
  DetachOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAccPtr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DetachOpAdaptor : public DetachOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DetachOpGenericAdaptor::DetachOpGenericAdaptor;
  DetachOpAdaptor(DetachOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DetachOp : public ::mlir::Op<DetachOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DetachOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DetachOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.detach");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getAccPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DetachOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DevicePtrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DevicePtrOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DevicePtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class DevicePtrOpGenericAdaptor : public detail::DevicePtrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DevicePtrOpGenericAdaptorBase;
public:
  DevicePtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DevicePtrOpAdaptor : public DevicePtrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DevicePtrOpGenericAdaptor::DevicePtrOpGenericAdaptor;
  DevicePtrOpAdaptor(DevicePtrOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DevicePtrOp : public ::mlir::Op<DevicePtrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DevicePtrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DevicePtrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.deviceptr");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DevicePtrOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp declarations
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
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
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

  ValueT getIfCond() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAsyncOperand() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getWaitDevnum() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getWaitOperands() {
    return getODSOperands(3);
  }

  RangeT getCopyinOperands() {
    return getODSOperands(4);
  }

  RangeT getCreateOperands() {
    return getODSOperands(5);
  }

  RangeT getCreateZeroOperands() {
    return getODSOperands(6);
  }

  RangeT getAttachOperands() {
    return getODSOperands(7);
  }

  RangeT getDataClauseOperands() {
    return getODSOperands(8);
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
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getWaitAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getWaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.enter_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::Operation::operand_range getDataClauseOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  ::mlir::MutableOperandRange getDataClauseOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setAsync(bool attrValue);
  void setWaitAttr(::mlir::UnitAttr attr);
  void setWait(bool attrValue);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::EnterDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp declarations
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
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getFinalizeAttr();
  bool getFinalize();
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

  ValueT getIfCond() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAsyncOperand() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getWaitDevnum() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getWaitOperands() {
    return getODSOperands(3);
  }

  RangeT getCopyoutOperands() {
    return getODSOperands(4);
  }

  RangeT getDeleteOperands() {
    return getODSOperands(5);
  }

  RangeT getDetachOperands() {
    return getODSOperands(6);
  }

  RangeT getDataClauseOperands() {
    return getODSOperands(7);
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
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("finalize"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getFinalizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getFinalizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getWaitAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getWaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.exit_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getDeleteOperands();
  ::mlir::Operation::operand_range getDetachOperands();
  ::mlir::Operation::operand_range getDataClauseOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getDeleteOperandsMutable();
  ::mlir::MutableOperandRange getDetachOperandsMutable();
  ::mlir::MutableOperandRange getDataClauseOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getFinalizeAttr();
  bool getFinalize();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setAsync(bool attrValue);
  void setWaitAttr(::mlir::UnitAttr attr);
  void setWait(bool attrValue);
  void setFinalizeAttr(::mlir::UnitAttr attr);
  void setFinalize(bool attrValue);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  ::mlir::Attribute removeFinalizeAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/bool finalize = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/bool finalize = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ExitDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::GetDevicePtrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GetDevicePtrOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GetDevicePtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class GetDevicePtrOpGenericAdaptor : public detail::GetDevicePtrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetDevicePtrOpGenericAdaptorBase;
public:
  GetDevicePtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetDevicePtrOpAdaptor : public GetDevicePtrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetDevicePtrOpGenericAdaptor::GetDevicePtrOpGenericAdaptor;
  GetDevicePtrOpAdaptor(GetDevicePtrOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GetDevicePtrOp : public ::mlir::Op<GetDevicePtrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetDevicePtrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetDevicePtrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.getdeviceptr");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::GetDevicePtrOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InitOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  InitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class InitOpGenericAdaptor : public detail::InitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InitOpGenericAdaptorBase;
public:
  InitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getDeviceTypeOperands() {
    return getODSOperands(0);
  }

  ValueT getDeviceNumOperand() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InitOpAdaptor : public InitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InitOpGenericAdaptor::InitOpGenericAdaptor;
  InitOpAdaptor(InitOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InitOp : public ::mlir::Op<InitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDeviceTypeOperands();
  ::mlir::Value getDeviceNumOperand();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::MutableOperandRange getDeviceTypeOperandsMutable();
  ::mlir::MutableOperandRange getDeviceNumOperandMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::InitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::KernelsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class KernelsOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  KernelsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class KernelsOpGenericAdaptor : public detail::KernelsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::KernelsOpGenericAdaptorBase;
public:
  KernelsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAsync() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getWaitOperands() {
    return getODSOperands(1);
  }

  ValueT getNumGangs() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getNumWorkers() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getVectorLength() {
    auto operands = getODSOperands(4);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(5);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getSelfCond() {
    auto operands = getODSOperands(6);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getCopyOperands() {
    return getODSOperands(7);
  }

  RangeT getCopyinOperands() {
    return getODSOperands(8);
  }

  RangeT getCopyinReadonlyOperands() {
    return getODSOperands(9);
  }

  RangeT getCopyoutOperands() {
    return getODSOperands(10);
  }

  RangeT getCopyoutZeroOperands() {
    return getODSOperands(11);
  }

  RangeT getCreateOperands() {
    return getODSOperands(12);
  }

  RangeT getCreateZeroOperands() {
    return getODSOperands(13);
  }

  RangeT getNoCreateOperands() {
    return getODSOperands(14);
  }

  RangeT getPresentOperands() {
    return getODSOperands(15);
  }

  RangeT getDevicePtrOperands() {
    return getODSOperands(16);
  }

  RangeT getAttachOperands() {
    return getODSOperands(17);
  }

  RangeT getDataClauseOperands() {
    return getODSOperands(18);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class KernelsOpAdaptor : public KernelsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using KernelsOpGenericAdaptor::KernelsOpGenericAdaptor;
  KernelsOpAdaptor(KernelsOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class KernelsOp : public ::mlir::Op<KernelsOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = KernelsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = KernelsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("asyncAttr"), ::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("selfAttr"), ::llvm::StringRef("waitAttr")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getDefaultAttrAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getDefaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getSelfAttrAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getSelfAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getWaitAttrAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getWaitAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.kernels");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getAsync();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Value getNumGangs();
  ::mlir::Value getNumWorkers();
  ::mlir::Value getVectorLength();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::TypedValue<::mlir::IntegerType> getSelfCond();
  ::mlir::Operation::operand_range getCopyOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCopyinReadonlyOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getCopyoutZeroOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getNoCreateOperands();
  ::mlir::Operation::operand_range getPresentOperands();
  ::mlir::Operation::operand_range getDevicePtrOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::Operation::operand_range getDataClauseOperands();
  ::mlir::MutableOperandRange getAsyncMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getNumGangsMutable();
  ::mlir::MutableOperandRange getNumWorkersMutable();
  ::mlir::MutableOperandRange getVectorLengthMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getSelfCondMutable();
  ::mlir::MutableOperandRange getCopyOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCopyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getNoCreateOperandsMutable();
  ::mlir::MutableOperandRange getPresentOperandsMutable();
  ::mlir::MutableOperandRange getDevicePtrOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  ::mlir::MutableOperandRange getDataClauseOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  void setAsyncAttrAttr(::mlir::UnitAttr attr);
  void setAsyncAttr(bool attrValue);
  void setWaitAttrAttr(::mlir::UnitAttr attr);
  void setWaitAttr(bool attrValue);
  void setSelfAttrAttr(::mlir::UnitAttr attr);
  void setSelfAttr(bool attrValue);
  void setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  void setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue);
  ::mlir::Attribute removeAsyncAttrAttr();
  ::mlir::Attribute removeWaitAttrAttr();
  ::mlir::Attribute removeSelfAttrAttr();
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::KernelsOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class LoopOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  LoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getCollapseAttr();
  ::std::optional<uint64_t> getCollapse();
  ::mlir::UnitAttr getSeqAttr();
  bool getSeq();
  ::mlir::UnitAttr getIndependentAttr();
  bool getIndependent();
  ::mlir::UnitAttr getAuto_Attr();
  bool getAuto_();
  ::mlir::UnitAttr getHasGangAttr();
  bool getHasGang();
  ::mlir::UnitAttr getHasWorkerAttr();
  bool getHasWorker();
  ::mlir::UnitAttr getHasVectorAttr();
  bool getHasVector();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::std::optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class LoopOpGenericAdaptor : public detail::LoopOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopOpGenericAdaptorBase;
public:
  LoopOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getGangNum() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getGangStatic() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getWorkerNum() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getVectorLength() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getTileOperands() {
    return getODSOperands(4);
  }

  RangeT getPrivateOperands() {
    return getODSOperands(5);
  }

  RangeT getReductionOperands() {
    return getODSOperands(6);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopOpAdaptor : public LoopOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopOpGenericAdaptor::LoopOpGenericAdaptor;
  LoopOpAdaptor(LoopOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class LoopOp : public ::mlir::Op<LoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("auto_"), ::llvm::StringRef("collapse"), ::llvm::StringRef("hasGang"), ::llvm::StringRef("hasVector"), ::llvm::StringRef("hasWorker"), ::llvm::StringRef("independent"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("seq")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAuto_AttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAuto_AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getCollapseAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getCollapseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getHasGangAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getHasGangAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getHasVectorAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getHasVectorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getHasWorkerAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getHasWorkerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getIndependentAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getIndependentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getReductionOpAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getReductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getSeqAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getSeqAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.loop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getGangNum();
  ::mlir::Value getGangStatic();
  ::mlir::Value getWorkerNum();
  ::mlir::Value getVectorLength();
  ::mlir::Operation::operand_range getTileOperands();
  ::mlir::Operation::operand_range getPrivateOperands();
  ::mlir::Operation::operand_range getReductionOperands();
  ::mlir::MutableOperandRange getGangNumMutable();
  ::mlir::MutableOperandRange getGangStaticMutable();
  ::mlir::MutableOperandRange getWorkerNumMutable();
  ::mlir::MutableOperandRange getVectorLengthMutable();
  ::mlir::MutableOperandRange getTileOperandsMutable();
  ::mlir::MutableOperandRange getPrivateOperandsMutable();
  ::mlir::MutableOperandRange getReductionOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getCollapseAttr();
  ::std::optional<uint64_t> getCollapse();
  ::mlir::UnitAttr getSeqAttr();
  bool getSeq();
  ::mlir::UnitAttr getIndependentAttr();
  bool getIndependent();
  ::mlir::UnitAttr getAuto_Attr();
  bool getAuto_();
  ::mlir::UnitAttr getHasGangAttr();
  bool getHasGang();
  ::mlir::UnitAttr getHasWorkerAttr();
  bool getHasWorker();
  ::mlir::UnitAttr getHasVectorAttr();
  bool getHasVector();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::std::optional<::mlir::acc::ReductionOp> getReductionOp();
  void setCollapseAttr(::mlir::IntegerAttr attr);
  void setCollapse(::std::optional<uint64_t> attrValue);
  void setSeqAttr(::mlir::UnitAttr attr);
  void setSeq(bool attrValue);
  void setIndependentAttr(::mlir::UnitAttr attr);
  void setIndependent(bool attrValue);
  void setAuto_Attr(::mlir::UnitAttr attr);
  void setAuto_(bool attrValue);
  void setHasGangAttr(::mlir::UnitAttr attr);
  void setHasGang(bool attrValue);
  void setHasWorkerAttr(::mlir::UnitAttr attr);
  void setHasWorker(bool attrValue);
  void setHasVectorAttr(::mlir::UnitAttr attr);
  void setHasVector(bool attrValue);
  void setReductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void setReductionOp(::std::optional<::mlir::acc::ReductionOp> attrValue);
  ::mlir::Attribute removeCollapseAttr();
  ::mlir::Attribute removeSeqAttr();
  ::mlir::Attribute removeIndependentAttr();
  ::mlir::Attribute removeAuto_Attr();
  ::mlir::Attribute removeHasGangAttr();
  ::mlir::Attribute removeHasWorkerAttr();
  ::mlir::Attribute removeHasVectorAttr();
  ::mlir::Attribute removeReductionOpAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, /*optional*/::mlir::UnitAttr hasGang, /*optional*/::mlir::UnitAttr hasWorker, /*optional*/::mlir::UnitAttr hasVector, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, /*optional*/bool hasGang, /*optional*/bool hasWorker, /*optional*/bool hasVector, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands);
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
  static StringRef getAutoAttrStrName() { return "auto"; }
  static StringRef getGangNumKeyword() { return "num"; }
  static StringRef getGangStaticKeyword() { return "static"; }
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::LoopOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::NoCreateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class NoCreateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  NoCreateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class NoCreateOpGenericAdaptor : public detail::NoCreateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NoCreateOpGenericAdaptorBase;
public:
  NoCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class NoCreateOpAdaptor : public NoCreateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NoCreateOpGenericAdaptor::NoCreateOpGenericAdaptor;
  NoCreateOpAdaptor(NoCreateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class NoCreateOp : public ::mlir::Op<NoCreateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NoCreateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NoCreateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.nocreate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::NoCreateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp declarations
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
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::std::optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
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

  ValueT getAsync() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getWaitOperands() {
    return getODSOperands(1);
  }

  ValueT getNumGangs() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getNumWorkers() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getVectorLength() {
    auto operands = getODSOperands(4);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(5);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getSelfCond() {
    auto operands = getODSOperands(6);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getReductionOperands() {
    return getODSOperands(7);
  }

  RangeT getCopyOperands() {
    return getODSOperands(8);
  }

  RangeT getCopyinOperands() {
    return getODSOperands(9);
  }

  RangeT getCopyinReadonlyOperands() {
    return getODSOperands(10);
  }

  RangeT getCopyoutOperands() {
    return getODSOperands(11);
  }

  RangeT getCopyoutZeroOperands() {
    return getODSOperands(12);
  }

  RangeT getCreateOperands() {
    return getODSOperands(13);
  }

  RangeT getCreateZeroOperands() {
    return getODSOperands(14);
  }

  RangeT getNoCreateOperands() {
    return getODSOperands(15);
  }

  RangeT getPresentOperands() {
    return getODSOperands(16);
  }

  RangeT getDevicePtrOperands() {
    return getODSOperands(17);
  }

  RangeT getAttachOperands() {
    return getODSOperands(18);
  }

  RangeT getGangPrivateOperands() {
    return getODSOperands(19);
  }

  RangeT getGangFirstPrivateOperands() {
    return getODSOperands(20);
  }

  RangeT getDataClauseOperands() {
    return getODSOperands(21);
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
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("asyncAttr"), ::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("selfAttr"), ::llvm::StringRef("waitAttr")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getDefaultAttrAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getDefaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getReductionOpAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getReductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSelfAttrAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSelfAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getWaitAttrAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getWaitAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getAsync();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Value getNumGangs();
  ::mlir::Value getNumWorkers();
  ::mlir::Value getVectorLength();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::TypedValue<::mlir::IntegerType> getSelfCond();
  ::mlir::Operation::operand_range getReductionOperands();
  ::mlir::Operation::operand_range getCopyOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCopyinReadonlyOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getCopyoutZeroOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getNoCreateOperands();
  ::mlir::Operation::operand_range getPresentOperands();
  ::mlir::Operation::operand_range getDevicePtrOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::Operation::operand_range getGangPrivateOperands();
  ::mlir::Operation::operand_range getGangFirstPrivateOperands();
  ::mlir::Operation::operand_range getDataClauseOperands();
  ::mlir::MutableOperandRange getAsyncMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getNumGangsMutable();
  ::mlir::MutableOperandRange getNumWorkersMutable();
  ::mlir::MutableOperandRange getVectorLengthMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getSelfCondMutable();
  ::mlir::MutableOperandRange getReductionOperandsMutable();
  ::mlir::MutableOperandRange getCopyOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCopyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getNoCreateOperandsMutable();
  ::mlir::MutableOperandRange getPresentOperandsMutable();
  ::mlir::MutableOperandRange getDevicePtrOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  ::mlir::MutableOperandRange getGangPrivateOperandsMutable();
  ::mlir::MutableOperandRange getGangFirstPrivateOperandsMutable();
  ::mlir::MutableOperandRange getDataClauseOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::std::optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  void setAsyncAttrAttr(::mlir::UnitAttr attr);
  void setAsyncAttr(bool attrValue);
  void setWaitAttrAttr(::mlir::UnitAttr attr);
  void setWaitAttr(bool attrValue);
  void setSelfAttrAttr(::mlir::UnitAttr attr);
  void setSelfAttr(bool attrValue);
  void setReductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void setReductionOp(::std::optional<::mlir::acc::ReductionOp> attrValue);
  void setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  void setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue);
  ::mlir::Attribute removeAsyncAttrAttr();
  ::mlir::Attribute removeWaitAttrAttr();
  ::mlir::Attribute removeSelfAttrAttr();
  ::mlir::Attribute removeReductionOpAttr();
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
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
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ParallelOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::PresentOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PresentOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  PresentOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
};
} // namespace detail
template <typename RangeT>
class PresentOpGenericAdaptor : public detail::PresentOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PresentOpGenericAdaptorBase;
public:
  PresentOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVarPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVarPtrPtr() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getBounds() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PresentOpAdaptor : public PresentOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PresentOpGenericAdaptor::PresentOpGenericAdaptor;
  PresentOpAdaptor(PresentOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class PresentOp : public ::mlir::Op<PresentOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::acc::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PresentOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PresentOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dataClause"), ::llvm::StringRef("implicit"), ::llvm::StringRef("name"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("structured")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDataClauseAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDataClauseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getImplicitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getImplicitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStructuredAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStructuredAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.present");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr();
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtrPtr();
  ::mlir::Operation::operand_range getBounds();
  ::mlir::MutableOperandRange getVarPtrMutable();
  ::mlir::MutableOperandRange getVarPtrPtrMutable();
  ::mlir::MutableOperandRange getBoundsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getAccPtr();
  ::mlir::acc::DataClauseAttr getDataClauseAttr();
  ::mlir::acc::DataClause getDataClause();
  ::mlir::BoolAttr getStructuredAttr();
  bool getStructured();
  ::mlir::BoolAttr getImplicitAttr();
  bool getImplicit();
  ::mlir::StringAttr getNameAttr();
  ::std::optional< ::llvm::StringRef > getName();
  void setDataClauseAttr(::mlir::acc::DataClauseAttr attr);
  void setDataClause(::mlir::acc::DataClause attrValue);
  void setStructuredAttr(::mlir::BoolAttr attr);
  void setStructured(bool attrValue);
  void setImplicitAttr(::mlir::BoolAttr attr);
  void setImplicit(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr);
  void setName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name);
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::PresentOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::SerialOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SerialOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SerialOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::std::optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class SerialOpGenericAdaptor : public detail::SerialOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SerialOpGenericAdaptorBase;
public:
  SerialOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAsync() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getWaitOperands() {
    return getODSOperands(1);
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getSelfCond() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getReductionOperands() {
    return getODSOperands(4);
  }

  RangeT getCopyOperands() {
    return getODSOperands(5);
  }

  RangeT getCopyinOperands() {
    return getODSOperands(6);
  }

  RangeT getCopyinReadonlyOperands() {
    return getODSOperands(7);
  }

  RangeT getCopyoutOperands() {
    return getODSOperands(8);
  }

  RangeT getCopyoutZeroOperands() {
    return getODSOperands(9);
  }

  RangeT getCreateOperands() {
    return getODSOperands(10);
  }

  RangeT getCreateZeroOperands() {
    return getODSOperands(11);
  }

  RangeT getNoCreateOperands() {
    return getODSOperands(12);
  }

  RangeT getPresentOperands() {
    return getODSOperands(13);
  }

  RangeT getDevicePtrOperands() {
    return getODSOperands(14);
  }

  RangeT getAttachOperands() {
    return getODSOperands(15);
  }

  RangeT getGangPrivateOperands() {
    return getODSOperands(16);
  }

  RangeT getGangFirstPrivateOperands() {
    return getODSOperands(17);
  }

  RangeT getDataClauseOperands() {
    return getODSOperands(18);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SerialOpAdaptor : public SerialOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SerialOpGenericAdaptor::SerialOpGenericAdaptor;
  SerialOpAdaptor(SerialOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SerialOp : public ::mlir::Op<SerialOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SerialOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SerialOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("asyncAttr"), ::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("selfAttr"), ::llvm::StringRef("waitAttr")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getDefaultAttrAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getDefaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getReductionOpAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getReductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSelfAttrAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSelfAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getWaitAttrAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getWaitAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.serial");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getAsync();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::TypedValue<::mlir::IntegerType> getSelfCond();
  ::mlir::Operation::operand_range getReductionOperands();
  ::mlir::Operation::operand_range getCopyOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCopyinReadonlyOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getCopyoutZeroOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getNoCreateOperands();
  ::mlir::Operation::operand_range getPresentOperands();
  ::mlir::Operation::operand_range getDevicePtrOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::Operation::operand_range getGangPrivateOperands();
  ::mlir::Operation::operand_range getGangFirstPrivateOperands();
  ::mlir::Operation::operand_range getDataClauseOperands();
  ::mlir::MutableOperandRange getAsyncMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getSelfCondMutable();
  ::mlir::MutableOperandRange getReductionOperandsMutable();
  ::mlir::MutableOperandRange getCopyOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCopyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getNoCreateOperandsMutable();
  ::mlir::MutableOperandRange getPresentOperandsMutable();
  ::mlir::MutableOperandRange getDevicePtrOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  ::mlir::MutableOperandRange getGangPrivateOperandsMutable();
  ::mlir::MutableOperandRange getGangFirstPrivateOperandsMutable();
  ::mlir::MutableOperandRange getDataClauseOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::std::optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::std::optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  void setAsyncAttrAttr(::mlir::UnitAttr attr);
  void setAsyncAttr(bool attrValue);
  void setWaitAttrAttr(::mlir::UnitAttr attr);
  void setWaitAttr(bool attrValue);
  void setSelfAttrAttr(::mlir::UnitAttr attr);
  void setSelfAttr(bool attrValue);
  void setReductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void setReductionOp(::std::optional<::mlir::acc::ReductionOp> attrValue);
  void setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  void setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue);
  ::mlir::Attribute removeAsyncAttrAttr();
  ::mlir::Attribute removeWaitAttrAttr();
  ::mlir::Attribute removeSelfAttrAttr();
  ::mlir::Attribute removeReductionOpAttr();
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
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
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::SerialOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ShutdownOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ShutdownOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ShutdownOpGenericAdaptor : public detail::ShutdownOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ShutdownOpGenericAdaptorBase;
public:
  ShutdownOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getDeviceTypeOperands() {
    return getODSOperands(0);
  }

  ValueT getDeviceNumOperand() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ShutdownOpAdaptor : public ShutdownOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ShutdownOpGenericAdaptor::ShutdownOpGenericAdaptor;
  ShutdownOpAdaptor(ShutdownOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ShutdownOp : public ::mlir::Op<ShutdownOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShutdownOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ShutdownOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.shutdown");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDeviceTypeOperands();
  ::mlir::Value getDeviceNumOperand();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::MutableOperandRange getDeviceTypeOperandsMutable();
  ::mlir::MutableOperandRange getDeviceNumOperandMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ShutdownOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp declarations
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
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
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
    return ::llvm::StringLiteral("acc.terminator");
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::TerminatorOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class UpdateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  UpdateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getIfPresentAttr();
  bool getIfPresent();
};
} // namespace detail
template <typename RangeT>
class UpdateOpGenericAdaptor : public detail::UpdateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::UpdateOpGenericAdaptorBase;
public:
  UpdateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAsyncOperand() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getWaitDevnum() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getWaitOperands() {
    return getODSOperands(3);
  }

  RangeT getDeviceTypeOperands() {
    return getODSOperands(4);
  }

  RangeT getHostOperands() {
    return getODSOperands(5);
  }

  RangeT getDeviceOperands() {
    return getODSOperands(6);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class UpdateOpAdaptor : public UpdateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using UpdateOpGenericAdaptor::UpdateOpGenericAdaptor;
  UpdateOpAdaptor(UpdateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class UpdateOp : public ::mlir::Op<UpdateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UpdateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = UpdateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("ifPresent"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIfPresentAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIfPresentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getWaitAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getWaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Operation::operand_range getDeviceTypeOperands();
  ::mlir::Operation::operand_range getHostOperands();
  ::mlir::Operation::operand_range getDeviceOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getDeviceTypeOperandsMutable();
  ::mlir::MutableOperandRange getHostOperandsMutable();
  ::mlir::MutableOperandRange getDeviceOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getIfPresentAttr();
  bool getIfPresent();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setAsync(bool attrValue);
  void setWaitAttr(::mlir::UnitAttr attr);
  void setWait(bool attrValue);
  void setIfPresentAttr(::mlir::UnitAttr attr);
  void setIfPresent(bool attrValue);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  ::mlir::Attribute removeIfPresentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::UpdateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WaitOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
};
} // namespace detail
template <typename RangeT>
class WaitOpGenericAdaptor : public detail::WaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WaitOpGenericAdaptorBase;
public:
  WaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getWaitOperands() {
    return getODSOperands(0);
  }

  ValueT getAsyncOperand() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getWaitDevnum() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getIfCond() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WaitOpAdaptor : public WaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WaitOpGenericAdaptor::WaitOpGenericAdaptor;
  WaitOpAdaptor(WaitOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WaitOp : public ::mlir::Op<WaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setAsync(bool attrValue);
  ::mlir::Attribute removeAsyncAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::WaitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp declarations
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

  RangeT getOperands() {
    return getODSOperands(0);
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
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<ParallelOp, LoopOp, SerialOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
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
    return ::llvm::StringLiteral("acc.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getOperands();
  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::YieldOp)


#endif  // GET_OP_CLASSES

