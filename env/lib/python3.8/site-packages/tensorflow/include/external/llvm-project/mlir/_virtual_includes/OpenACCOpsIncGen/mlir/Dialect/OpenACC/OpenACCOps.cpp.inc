/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::acc::AttachOp,
::mlir::acc::CopyinOp,
::mlir::acc::CopyoutOp,
::mlir::acc::CreateOp,
::mlir::acc::DataBoundsOp,
::mlir::acc::DataOp,
::mlir::acc::DeleteOp,
::mlir::acc::DetachOp,
::mlir::acc::DevicePtrOp,
::mlir::acc::EnterDataOp,
::mlir::acc::ExitDataOp,
::mlir::acc::GetDevicePtrOp,
::mlir::acc::InitOp,
::mlir::acc::KernelsOp,
::mlir::acc::LoopOp,
::mlir::acc::NoCreateOp,
::mlir::acc::ParallelOp,
::mlir::acc::PresentOp,
::mlir::acc::SerialOp,
::mlir::acc::ShutdownOp,
::mlir::acc::TerminatorOp,
::mlir::acc::UpdateOp,
::mlir::acc::WaitOp,
::mlir::acc::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace acc {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::acc::PointerLikeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PointerLikeType instance, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::acc::DataBoundsType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Type for representing acc data clause bounds information, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::acc::DataClauseAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: data clauses supported by OpenACC";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::acc::ClauseDefaultValueAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: DefaultValue Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::acc::ReductionOpAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: built-in reduction operations supported by OpenACC";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_OpenACCOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::AttachOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AttachOpGenericAdaptorBase::AttachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.attach", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AttachOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AttachOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr AttachOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr AttachOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AttachOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause AttachOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr AttachOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AttachOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool AttachOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr AttachOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AttachOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool AttachOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr AttachOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AttachOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > AttachOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
AttachOpAdaptor::AttachOpAdaptor(AttachOp op) : AttachOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AttachOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.attach' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == AttachOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AttachOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AttachOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AttachOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AttachOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.attach' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.attach' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.attach' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.attach' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.attach' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AttachOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range AttachOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> AttachOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> AttachOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range AttachOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AttachOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange AttachOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange AttachOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> AttachOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AttachOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> AttachOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr AttachOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause AttachOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr AttachOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool AttachOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr AttachOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool AttachOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr AttachOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > AttachOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void AttachOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void AttachOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AttachOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void AttachOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void AttachOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void AttachOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void AttachOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void AttachOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute AttachOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void AttachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void AttachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AttachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void AttachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AttachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AttachOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_attach));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult AttachOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AttachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AttachOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AttachOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_attach)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::AttachOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::CopyinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CopyinOpGenericAdaptorBase::CopyinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.copyin", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CopyinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CopyinOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CopyinOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr CopyinOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyinOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause CopyinOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyinOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, CopyinOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool CopyinOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyinOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyinOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool CopyinOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr CopyinOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyinOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > CopyinOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
CopyinOpAdaptor::CopyinOpAdaptor(CopyinOp op) : CopyinOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CopyinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.copyin' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CopyinOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CopyinOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CopyinOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CopyinOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CopyinOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.copyin' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.copyin' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.copyin' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.copyin' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.copyin' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CopyinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CopyinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CopyinOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CopyinOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range CopyinOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CopyinOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyinOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyinOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CopyinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopyinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CopyinOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr CopyinOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause CopyinOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyinOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool CopyinOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyinOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool CopyinOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr CopyinOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > CopyinOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void CopyinOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void CopyinOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void CopyinOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void CopyinOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void CopyinOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void CopyinOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void CopyinOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CopyinOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute CopyinOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void CopyinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void CopyinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopyinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void CopyinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopyinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CopyinOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_copyin));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult CopyinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CopyinOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CopyinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CopyinOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_copyin)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::CopyinOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::CopyoutOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CopyoutOpGenericAdaptorBase::CopyoutOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.copyout", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CopyoutOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CopyoutOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CopyoutOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr CopyoutOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyoutOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause CopyoutOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyoutOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, CopyoutOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool CopyoutOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyoutOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyoutOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool CopyoutOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr CopyoutOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyoutOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > CopyoutOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
CopyoutOpAdaptor::CopyoutOpAdaptor(CopyoutOp op) : CopyoutOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CopyoutOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.copyout' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CopyoutOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CopyoutOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CopyoutOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CopyoutOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CopyoutOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.copyout' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.copyout' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.copyout' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.copyout' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.copyout' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CopyoutOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CopyoutOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CopyoutOp::getVarPtr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CopyoutOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range CopyoutOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CopyoutOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyoutOp::getAccPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyoutOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CopyoutOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopyoutOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::acc::DataClauseAttr CopyoutOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause CopyoutOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyoutOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool CopyoutOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr CopyoutOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool CopyoutOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr CopyoutOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > CopyoutOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void CopyoutOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void CopyoutOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void CopyoutOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void CopyoutOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void CopyoutOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void CopyoutOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void CopyoutOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CopyoutOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute CopyoutOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void CopyoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
}

void CopyoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopyoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
}

void CopyoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopyoutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CopyoutOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_copyout));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult CopyoutOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CopyoutOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CopyoutOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand accPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accPtrOperands(accPtrRawOperands);  ::llvm::SMLoc accPtrOperandsLoc;
  (void)accPtrOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrOperands;
  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrTypes;
  if (parser.parseKeyword("accPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  accPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool boundsClause = false;
  bool toClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("to"))) {

  if (toClause) {
    return parser.emitError(parser.getNameLoc())
          << "`to` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  toClause = true;
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(varPtrOperands.size()), 1, static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accPtrOperands, accPtrTypes, accPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CopyoutOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "accPtr";
  _odsPrinter << "(";
  _odsPrinter << getAccPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  if (false || getVarPtr()) {
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_copyout)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::CopyoutOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::CreateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateOpGenericAdaptorBase::CreateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.create", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CreateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr CreateOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CreateOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause CreateOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr CreateOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, CreateOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool CreateOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr CreateOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CreateOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool CreateOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr CreateOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CreateOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > CreateOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
CreateOpAdaptor::CreateOpAdaptor(CreateOp op) : CreateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CreateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.create' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CreateOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CreateOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CreateOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CreateOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CreateOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.create' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.create' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.create' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.create' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.create' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CreateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CreateOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CreateOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range CreateOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CreateOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CreateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> CreateOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr CreateOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause CreateOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr CreateOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool CreateOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr CreateOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool CreateOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr CreateOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > CreateOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void CreateOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void CreateOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void CreateOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void CreateOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void CreateOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void CreateOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void CreateOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CreateOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute CreateOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void CreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void CreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void CreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CreateOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_create));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult CreateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_create)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::CreateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataBoundsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DataBoundsOpGenericAdaptorBase::DataBoundsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.bounds", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DataBoundsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DataBoundsOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DataBoundsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr DataBoundsOpGenericAdaptorBase::getStrideInBytesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DataBoundsOp::getStrideInBytesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DataBoundsOpGenericAdaptorBase::getStrideInBytes() {
  auto attr = getStrideInBytesAttr();
  return attr.getValue();
}

} // namespace detail
DataBoundsOpAdaptor::DataBoundsOpAdaptor(DataBoundsOp op) : DataBoundsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DataBoundsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.bounds' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DataBoundsOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strideInBytes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DataBoundsOp::getStrideInBytesAttrName(*odsOpName)) {
      tblgen_strideInBytes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'acc.bounds' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strideInBytes && !((tblgen_strideInBytes.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.bounds' op ""attribute 'strideInBytes' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DataBoundsOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DataBoundsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DataBoundsOp::getLowerbound() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value DataBoundsOp::getUpperbound() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value DataBoundsOp::getExtent() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value DataBoundsOp::getStride() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value DataBoundsOp::getStartIdx() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::MutableOperandRange DataBoundsOp::getLowerboundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataBoundsOp::getUpperboundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataBoundsOp::getExtentMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataBoundsOp::getStrideMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataBoundsOp::getStartIdxMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DataBoundsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DataBoundsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::DataBoundsType> DataBoundsOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::DataBoundsType>>(*getODSResults(0).begin());
}

::mlir::BoolAttr DataBoundsOp::getStrideInBytesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStrideInBytesAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DataBoundsOp::getStrideInBytes() {
  auto attr = getStrideInBytesAttr();
  return attr.getValue();
}

void DataBoundsOp::setStrideInBytesAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStrideInBytesAttrName(), attr);
}

void DataBoundsOp::setStrideInBytes(bool attrValue) {
  (*this)->setAttr(getStrideInBytesAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DataBoundsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, ::mlir::BoolAttr strideInBytes, /*optional*/::mlir::Value startIdx) {
  if (lowerbound)
    odsState.addOperands(lowerbound);
  if (upperbound)
    odsState.addOperands(upperbound);
  if (extent)
    odsState.addOperands(extent);
  if (stride)
    odsState.addOperands(stride);
  if (startIdx)
    odsState.addOperands(startIdx);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(lowerbound ? 1 : 0), (upperbound ? 1 : 0), (extent ? 1 : 0), (stride ? 1 : 0), (startIdx ? 1 : 0)}));
  if (strideInBytes) {
    odsState.addAttribute(getStrideInBytesAttrName(odsState.name), strideInBytes);
  }
  odsState.addTypes(result);
}

void DataBoundsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, ::mlir::BoolAttr strideInBytes, /*optional*/::mlir::Value startIdx) {
  if (lowerbound)
    odsState.addOperands(lowerbound);
  if (upperbound)
    odsState.addOperands(upperbound);
  if (extent)
    odsState.addOperands(extent);
  if (stride)
    odsState.addOperands(stride);
  if (startIdx)
    odsState.addOperands(startIdx);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(lowerbound ? 1 : 0), (upperbound ? 1 : 0), (extent ? 1 : 0), (stride ? 1 : 0), (startIdx ? 1 : 0)}));
  if (strideInBytes) {
    odsState.addAttribute(getStrideInBytesAttrName(odsState.name), strideInBytes);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DataBoundsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, bool strideInBytes, /*optional*/::mlir::Value startIdx) {
  if (lowerbound)
    odsState.addOperands(lowerbound);
  if (upperbound)
    odsState.addOperands(upperbound);
  if (extent)
    odsState.addOperands(extent);
  if (stride)
    odsState.addOperands(stride);
  if (startIdx)
    odsState.addOperands(startIdx);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(lowerbound ? 1 : 0), (upperbound ? 1 : 0), (extent ? 1 : 0), (stride ? 1 : 0), (startIdx ? 1 : 0)}));
  odsState.addAttribute(getStrideInBytesAttrName(odsState.name), odsBuilder.getBoolAttr(strideInBytes));
  odsState.addTypes(result);
}

void DataBoundsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value lowerbound, /*optional*/::mlir::Value upperbound, /*optional*/::mlir::Value extent, /*optional*/::mlir::Value stride, bool strideInBytes, /*optional*/::mlir::Value startIdx) {
  if (lowerbound)
    odsState.addOperands(lowerbound);
  if (upperbound)
    odsState.addOperands(upperbound);
  if (extent)
    odsState.addOperands(extent);
  if (stride)
    odsState.addOperands(stride);
  if (startIdx)
    odsState.addOperands(startIdx);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(lowerbound ? 1 : 0), (upperbound ? 1 : 0), (extent ? 1 : 0), (stride ? 1 : 0), (startIdx ? 1 : 0)}));
  odsState.addAttribute(getStrideInBytesAttrName(odsState.name), odsBuilder.getBoolAttr(strideInBytes));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DataBoundsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DataBoundsOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult DataBoundsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strideInBytes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStrideInBytesAttrName()) {
      tblgen_strideInBytes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_strideInBytes, "strideInBytes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DataBoundsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DataBoundsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lowerboundOperands;
  ::llvm::SMLoc lowerboundOperandsLoc;
  (void)lowerboundOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> lowerboundTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> upperboundOperands;
  ::llvm::SMLoc upperboundOperandsLoc;
  (void)upperboundOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> upperboundTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> extentOperands;
  ::llvm::SMLoc extentOperandsLoc;
  (void)extentOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> extentTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> strideOperands;
  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> strideTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> startIdxOperands;
  ::llvm::SMLoc startIdxOperandsLoc;
  (void)startIdxOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> startIdxTypes;
  bool lowerboundClause = false;
  bool upperboundClause = false;
  bool extentClause = false;
  bool strideClause = false;
  bool startIdxClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("lowerbound"))) {

  if (lowerboundClause) {
    return parser.emitError(parser.getNameLoc())
          << "`lowerbound` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  lowerboundClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    lowerboundOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      lowerboundOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      lowerboundTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("upperbound"))) {

  if (upperboundClause) {
    return parser.emitError(parser.getNameLoc())
          << "`upperbound` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  upperboundClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    upperboundOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      upperboundOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      upperboundTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("extent"))) {

  if (extentClause) {
    return parser.emitError(parser.getNameLoc())
          << "`extent` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  extentClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    extentOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      extentOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      extentTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("stride"))) {

  if (strideClause) {
    return parser.emitError(parser.getNameLoc())
          << "`stride` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  strideClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    strideOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      strideOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      strideTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("startIdx"))) {

  if (startIdxClause) {
    return parser.emitError(parser.getNameLoc())
          << "`startIdx` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  startIdxClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    startIdxOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      startIdxOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      startIdxTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(lowerboundOperands.size()), static_cast<int32_t>(upperboundOperands.size()), static_cast<int32_t>(extentOperands.size()), static_cast<int32_t>(strideOperands.size()), static_cast<int32_t>(startIdxOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(lowerboundOperands, lowerboundTypes, lowerboundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperboundOperands, upperboundTypes, upperboundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(extentOperands, extentTypes, extentOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(startIdxOperands, startIdxTypes, startIdxOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DataBoundsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getLowerbound()) {
  _odsPrinter << ' ' << "lowerbound";
  _odsPrinter << "(";
  if (::mlir::Value value = getLowerbound())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getLowerbound() ? ::llvm::ArrayRef<::mlir::Type>(getLowerbound().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getUpperbound()) {
  _odsPrinter << ' ' << "upperbound";
  _odsPrinter << "(";
  if (::mlir::Value value = getUpperbound())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getUpperbound() ? ::llvm::ArrayRef<::mlir::Type>(getUpperbound().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getExtent()) {
  _odsPrinter << ' ' << "extent";
  _odsPrinter << "(";
  if (::mlir::Value value = getExtent())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getExtent() ? ::llvm::ArrayRef<::mlir::Type>(getExtent().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getStride()) {
  _odsPrinter << ' ' << "stride";
  _odsPrinter << "(";
  if (::mlir::Value value = getStride())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getStride() ? ::llvm::ArrayRef<::mlir::Type>(getStride().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getStartIdx()) {
  _odsPrinter << ' ' << "startIdx";
  _odsPrinter << "(";
  if (::mlir::Value value = getStartIdx())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getStartIdx() ? ::llvm::ArrayRef<::mlir::Type>(getStartIdx().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStrideInBytesAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("strideInBytes");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void DataBoundsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DataBoundsOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DataOpGenericAdaptorBase::DataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.data", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DataOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::ClauseDefaultValueAttr DataOpGenericAdaptorBase::getDefaultAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DataOp::getDefaultAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
  return attr;
}

::std::optional<::mlir::acc::ClauseDefaultValue> DataOpGenericAdaptorBase::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &DataOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DataOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DataOpAdaptor::DataOpAdaptor(DataOp op) : DataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DataOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DataOp::getDefaultAttrAttrName(*odsOpName)) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 13)
      return emitError(loc, "'acc.data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 13 "
                "elements, but got ") << numElements;
  }

  if (tblgen_defaultAttr && !((tblgen_defaultAttr.isa<::mlir::acc::ClauseDefaultValueAttr>())))
    return emitError(loc, "'acc.data' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> DataOp::getIfCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range DataOp::getCopyOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DataOp::getCopyinOperands() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range DataOp::getCopyinReadonlyOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range DataOp::getCopyoutOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range DataOp::getCopyoutZeroOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range DataOp::getCreateOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range DataOp::getCreateZeroOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range DataOp::getNoCreateOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range DataOp::getPresentOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range DataOp::getDeviceptrOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range DataOp::getAttachOperands() {
  return getODSOperands(11);
}

::mlir::Operation::operand_range DataOp::getDataClauseOperands() {
  return getODSOperands(12);
}

::mlir::MutableOperandRange DataOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCopyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCopyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCopyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCopyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCopyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getCreateZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getNoCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getPresentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getDeviceptrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getAttachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getDataClauseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(12);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(12u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &DataOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::acc::ClauseDefaultValueAttr DataOp::getDefaultAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDefaultAttrAttrName()).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
}

::std::optional<::mlir::acc::ClauseDefaultValue> DataOp::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

void DataOp::setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr) {
  (*this)->setAttr(getDefaultAttrAttrName(), attr);
}

void DataOp::setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDefaultAttrAttrName(), ::mlir::acc::ClauseDefaultValueAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getDefaultAttrAttrName());
}

::mlir::Attribute DataOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr(getDefaultAttrAttrName());
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(deviceptrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(deviceptrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(deviceptrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(deviceptrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDefaultAttrAttrName()) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 13)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 13 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps3(*this, tblgen_defaultAttr, "defaultAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup12 = getODSOperands(12);

    for (auto v : valueGroup12) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dataClauseOperandsOperands;
  ::llvm::SMLoc dataClauseOperandsOperandsLoc;
  (void)dataClauseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> dataClauseOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyOperandsOperands;
  ::llvm::SMLoc copyOperandsOperandsLoc;
  (void)copyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinReadonlyOperandsOperands;
  ::llvm::SMLoc copyinReadonlyOperandsOperandsLoc;
  (void)copyinReadonlyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinReadonlyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutZeroOperandsOperands;
  ::llvm::SMLoc copyoutZeroOperandsOperandsLoc;
  (void)copyoutZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> noCreateOperandsOperands;
  ::llvm::SMLoc noCreateOperandsOperandsLoc;
  (void)noCreateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> noCreateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> presentOperandsOperands;
  ::llvm::SMLoc presentOperandsOperandsLoc;
  (void)presentOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> presentOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceptrOperandsOperands;
  ::llvm::SMLoc deviceptrOperandsOperandsLoc;
  (void)deviceptrOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceptrOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool dataOperandsClause = false;
  bool copyClause = false;
  bool copyinClause = false;
  bool copyin_readonlyClause = false;
  bool copyoutClause = false;
  bool copyout_zeroClause = false;
  bool createClause = false;
  bool create_zeroClause = false;
  bool no_createClause = false;
  bool presentClause = false;
  bool deviceptrClause = false;
  bool attachClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("dataOperands"))) {

  if (dataOperandsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`dataOperands` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dataOperandsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  dataClauseOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dataClauseOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(dataClauseOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copy"))) {

  if (copyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copy` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin"))) {

  if (copyinClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyinClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin_readonly"))) {

  if (copyin_readonlyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin_readonly` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyin_readonlyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinReadonlyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinReadonlyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinReadonlyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout"))) {

  if (copyoutClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyoutClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout_zero"))) {

  if (copyout_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyout_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create"))) {

  if (createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create_zero"))) {

  if (create_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  create_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("no_create"))) {

  if (no_createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`no_create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  no_createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  noCreateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(noCreateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(noCreateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("present"))) {

  if (presentClause) {
    return parser.emitError(parser.getNameLoc())
          << "`present` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  presentClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  presentOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(presentOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(presentOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("deviceptr"))) {

  if (deviceptrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`deviceptr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deviceptrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceptrOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceptrOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceptrOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("attach"))) {

  if (attachClause) {
    return parser.emitError(parser.getNameLoc())
          << "`attach` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  attachClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(copyOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(copyinReadonlyOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(copyoutZeroOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(noCreateOperandsOperands.size()), static_cast<int32_t>(presentOperandsOperands.size()), static_cast<int32_t>(deviceptrOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size()), static_cast<int32_t>(dataClauseOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyOperandsOperands, copyOperandsTypes, copyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinReadonlyOperandsOperands, copyinReadonlyOperandsTypes, copyinReadonlyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutZeroOperandsOperands, copyoutZeroOperandsTypes, copyoutZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(noCreateOperandsOperands, noCreateOperandsTypes, noCreateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(presentOperandsOperands, presentOperandsTypes, presentOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceptrOperandsOperands, deviceptrOperandsTypes, deviceptrOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataClauseOperandsOperands, dataClauseOperandsTypes, dataClauseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getDataClauseOperands().size()) {
  _odsPrinter << ' ' << "dataOperands";
  _odsPrinter << "(";
  _odsPrinter << getDataClauseOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDataClauseOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyOperands().size()) {
  _odsPrinter << ' ' << "copy";
  _odsPrinter << "(";
  _odsPrinter << getCopyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinOperands().size()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << getCopyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinReadonlyOperands().size()) {
  _odsPrinter << ' ' << "copyin_readonly";
  _odsPrinter << "(";
  _odsPrinter << getCopyinReadonlyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinReadonlyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutOperands().size()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutZeroOperands().size()) {
  _odsPrinter << ' ' << "copyout_zero";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateOperands().size()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << getCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateZeroOperands().size()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << getCreateZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getNoCreateOperands().size()) {
  _odsPrinter << ' ' << "no_create";
  _odsPrinter << "(";
  _odsPrinter << getNoCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getNoCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getPresentOperands().size()) {
  _odsPrinter << ' ' << "present";
  _odsPrinter << "(";
  _odsPrinter << getPresentOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getPresentOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDeviceptrOperands().size()) {
  _odsPrinter << ' ' << "deviceptr";
  _odsPrinter << "(";
  _odsPrinter << getDeviceptrOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDeviceptrOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAttachOperands().size()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << getAttachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getAttachOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DeleteOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DeleteOpGenericAdaptorBase::DeleteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.delete", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DeleteOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DeleteOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DeleteOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr DeleteOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DeleteOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause DeleteOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr DeleteOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DeleteOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DeleteOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr DeleteOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DeleteOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DeleteOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr DeleteOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DeleteOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > DeleteOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
DeleteOpAdaptor::DeleteOpAdaptor(DeleteOp op) : DeleteOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DeleteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.delete' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DeleteOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DeleteOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == DeleteOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == DeleteOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DeleteOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.delete' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.delete' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.delete' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.delete' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.delete' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DeleteOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DeleteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DeleteOp::getVarPtr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DeleteOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range DeleteOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange DeleteOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DeleteOp::getAccPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DeleteOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DeleteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeleteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::acc::DataClauseAttr DeleteOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause DeleteOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr DeleteOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DeleteOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr DeleteOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DeleteOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr DeleteOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > DeleteOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void DeleteOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void DeleteOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void DeleteOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void DeleteOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DeleteOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void DeleteOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DeleteOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void DeleteOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute DeleteOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void DeleteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
}

void DeleteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeleteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
}

void DeleteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeleteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DeleteOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_delete));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult DeleteOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DeleteOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DeleteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand accPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accPtrOperands(accPtrRawOperands);  ::llvm::SMLoc accPtrOperandsLoc;
  (void)accPtrOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrOperands;
  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrTypes;
  if (parser.parseKeyword("accPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  accPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool boundsClause = false;
  bool toClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("to"))) {

  if (toClause) {
    return parser.emitError(parser.getNameLoc())
          << "`to` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  toClause = true;
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(varPtrOperands.size()), 1, static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accPtrOperands, accPtrTypes, accPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeleteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "accPtr";
  _odsPrinter << "(";
  _odsPrinter << getAccPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  if (false || getVarPtr()) {
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_delete)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DeleteOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DetachOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DetachOpGenericAdaptorBase::DetachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.detach", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DetachOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DetachOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DetachOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr DetachOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DetachOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause DetachOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr DetachOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DetachOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DetachOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr DetachOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DetachOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DetachOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr DetachOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DetachOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > DetachOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
DetachOpAdaptor::DetachOpAdaptor(DetachOp op) : DetachOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DetachOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.detach' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DetachOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DetachOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == DetachOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == DetachOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DetachOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.detach' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.detach' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.detach' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.detach' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.detach' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DetachOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DetachOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DetachOp::getVarPtr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DetachOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range DetachOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange DetachOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DetachOp::getAccPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DetachOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DetachOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DetachOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::acc::DataClauseAttr DetachOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause DetachOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr DetachOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DetachOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr DetachOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DetachOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr DetachOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > DetachOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void DetachOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void DetachOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void DetachOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void DetachOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DetachOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void DetachOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DetachOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void DetachOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute DetachOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void DetachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
}

void DetachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DetachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
}

void DetachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value varPtr, ::mlir::Value accPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  if (varPtr)
    odsState.addOperands(varPtr);
  odsState.addOperands(accPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(varPtr ? 1 : 0), 1, static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DetachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DetachOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_detach));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult DetachOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DetachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DetachOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand accPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accPtrOperands(accPtrRawOperands);  ::llvm::SMLoc accPtrOperandsLoc;
  (void)accPtrOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrOperands;
  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrTypes;
  if (parser.parseKeyword("accPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  accPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool boundsClause = false;
  bool toClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("to"))) {

  if (toClause) {
    return parser.emitError(parser.getNameLoc())
          << "`to` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  toClause = true;
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(varPtrOperands.size()), 1, static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accPtrOperands, accPtrTypes, accPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DetachOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "accPtr";
  _odsPrinter << "(";
  _odsPrinter << getAccPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  if (false || getVarPtr()) {
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_detach)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DetachOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DevicePtrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DevicePtrOpGenericAdaptorBase::DevicePtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.deviceptr", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DevicePtrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DevicePtrOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DevicePtrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr DevicePtrOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DevicePtrOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause DevicePtrOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr DevicePtrOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DevicePtrOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DevicePtrOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr DevicePtrOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DevicePtrOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool DevicePtrOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr DevicePtrOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DevicePtrOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > DevicePtrOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
DevicePtrOpAdaptor::DevicePtrOpAdaptor(DevicePtrOp op) : DevicePtrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DevicePtrOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.deviceptr' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DevicePtrOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DevicePtrOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == DevicePtrOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == DevicePtrOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DevicePtrOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.deviceptr' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.deviceptr' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.deviceptr' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.deviceptr' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.deviceptr' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DevicePtrOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DevicePtrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DevicePtrOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DevicePtrOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range DevicePtrOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange DevicePtrOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DevicePtrOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DevicePtrOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DevicePtrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DevicePtrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> DevicePtrOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr DevicePtrOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause DevicePtrOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr DevicePtrOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DevicePtrOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr DevicePtrOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool DevicePtrOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr DevicePtrOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > DevicePtrOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void DevicePtrOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void DevicePtrOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void DevicePtrOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void DevicePtrOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DevicePtrOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void DevicePtrOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void DevicePtrOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void DevicePtrOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute DevicePtrOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void DevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void DevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void DevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DevicePtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DevicePtrOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_deviceptr));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult DevicePtrOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DevicePtrOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DevicePtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DevicePtrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_deviceptr)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DevicePtrOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EnterDataOpGenericAdaptorBase::EnterDataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.enter_data", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EnterDataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, EnterDataOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr EnterDataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr EnterDataOpGenericAdaptorBase::getAsyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, EnterDataOp::getAsyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool EnterDataOpGenericAdaptorBase::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr EnterDataOpGenericAdaptorBase::getWaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, EnterDataOp::getWaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool EnterDataOpGenericAdaptorBase::getWait() {
  auto attr = getWaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
EnterDataOpAdaptor::EnterDataOpAdaptor(EnterDataOp op) : EnterDataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult EnterDataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.enter_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == EnterDataOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == EnterDataOp::getAsyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == EnterDataOp::getWaitAttrName(*odsOpName)) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 9)
      return emitError(loc, "'acc.enter_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 9 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.enter_data' op ""attribute 'async' failed to satisfy constraint: unit attribute");

  if (tblgen_wait && !((tblgen_wait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.enter_data' op ""attribute 'wait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EnterDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range EnterDataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> EnterDataOp::getIfCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Value EnterDataOp::getAsyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value EnterDataOp::getWaitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range EnterDataOp::getWaitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range EnterDataOp::getCopyinOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range EnterDataOp::getCreateOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range EnterDataOp::getCreateZeroOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range EnterDataOp::getAttachOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range EnterDataOp::getDataClauseOperands() {
  return getODSOperands(8);
}

::mlir::MutableOperandRange EnterDataOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getAsyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getWaitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getCopyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getCreateZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getAttachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getDataClauseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> EnterDataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EnterDataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr EnterDataOp::getAsyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr EnterDataOp::getWaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getWaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::getWait() {
  auto attr = getWaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void EnterDataOp::setAsyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrName(), attr);
}

void EnterDataOp::setAsync(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrName());
}

void EnterDataOp::setWaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWaitAttrName(), attr);
}

void EnterDataOp::setWait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWaitAttrName());
}

::mlir::Attribute EnterDataOp::removeAsyncAttr() {
  return (*this)->removeAttr(getAsyncAttrName());
}

::mlir::Attribute EnterDataOp::removeWaitAttr() {
  return (*this)->removeAttr(getWaitAttrName());
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), wait);
  }
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), wait);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), ((wait) ? odsBuilder.getUnitAttr() : nullptr));
  }
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), ((wait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EnterDataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getWaitAttrName()) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 9)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 9 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_async, "async")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_wait, "wait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EnterDataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult EnterDataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dataClauseOperandsOperands;
  ::llvm::SMLoc dataClauseOperandsOperandsLoc;
  (void)dataClauseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> dataClauseOperandsTypes;
  bool ifClause = false;
  bool asyncClause = false;
  bool wait_devnumClause = false;
  bool waitClause = false;
  bool copyinClause = false;
  bool createClause = false;
  bool create_zeroClause = false;
  bool attachClause = false;
  bool dataOperandsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {

  if (wait_devnumClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait_devnum` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  wait_devnumClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait"))) {

  if (waitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  waitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin"))) {

  if (copyinClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyinClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create"))) {

  if (createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create_zero"))) {

  if (create_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  create_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("attach"))) {

  if (attachClause) {
    return parser.emitError(parser.getNameLoc())
          << "`attach` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  attachClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("dataOperands"))) {

  if (dataOperandsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`dataOperands` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dataOperandsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  dataClauseOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dataClauseOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(dataClauseOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size()), static_cast<int32_t>(dataClauseOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataClauseOperandsOperands, dataClauseOperandsTypes, dataClauseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EnterDataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getAsyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(getAsyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = getWaitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getWaitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(getWaitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitOperands().size()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << getWaitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getWaitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinOperands().size()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << getCopyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateOperands().size()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << getCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateZeroOperands().size()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << getCreateZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAttachOperands().size()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << getAttachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getAttachOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDataClauseOperands().size()) {
  _odsPrinter << ' ' << "dataOperands";
  _odsPrinter << "(";
  _odsPrinter << getDataClauseOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDataClauseOperands().getTypes();
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("async");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("wait");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::EnterDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExitDataOpGenericAdaptorBase::ExitDataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.exit_data", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExitDataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExitDataOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ExitDataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ExitDataOpGenericAdaptorBase::getAsyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ExitDataOp::getAsyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ExitDataOpGenericAdaptorBase::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOpGenericAdaptorBase::getWaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ExitDataOp::getWaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ExitDataOpGenericAdaptorBase::getWait() {
  auto attr = getWaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOpGenericAdaptorBase::getFinalizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ExitDataOp::getFinalizeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ExitDataOpGenericAdaptorBase::getFinalize() {
  auto attr = getFinalizeAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
ExitDataOpAdaptor::ExitDataOpAdaptor(ExitDataOp op) : ExitDataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExitDataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_finalize;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.exit_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ExitDataOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ExitDataOp::getAsyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ExitDataOp::getFinalizeAttrName(*odsOpName)) {
      tblgen_finalize = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ExitDataOp::getWaitAttrName(*odsOpName)) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 8)
      return emitError(loc, "'acc.exit_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 8 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.exit_data' op ""attribute 'async' failed to satisfy constraint: unit attribute");

  if (tblgen_wait && !((tblgen_wait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.exit_data' op ""attribute 'wait' failed to satisfy constraint: unit attribute");

  if (tblgen_finalize && !((tblgen_finalize.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.exit_data' op ""attribute 'finalize' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExitDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ExitDataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> ExitDataOp::getIfCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Value ExitDataOp::getAsyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value ExitDataOp::getWaitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range ExitDataOp::getWaitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ExitDataOp::getCopyoutOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range ExitDataOp::getDeleteOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range ExitDataOp::getDetachOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range ExitDataOp::getDataClauseOperands() {
  return getODSOperands(7);
}

::mlir::MutableOperandRange ExitDataOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getAsyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getWaitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getCopyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getDeleteOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getDetachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getDataClauseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ExitDataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExitDataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr ExitDataOp::getAsyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOp::getWaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getWaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::getWait() {
  auto attr = getWaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOp::getFinalizeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFinalizeAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::getFinalize() {
  auto attr = getFinalizeAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void ExitDataOp::setAsyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrName(), attr);
}

void ExitDataOp::setAsync(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrName());
}

void ExitDataOp::setWaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWaitAttrName(), attr);
}

void ExitDataOp::setWait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWaitAttrName());
}

void ExitDataOp::setFinalizeAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getFinalizeAttrName(), attr);
}

void ExitDataOp::setFinalize(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getFinalizeAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getFinalizeAttrName());
}

::mlir::Attribute ExitDataOp::removeAsyncAttr() {
  return (*this)->removeAttr(getAsyncAttrName());
}

::mlir::Attribute ExitDataOp::removeWaitAttr() {
  return (*this)->removeAttr(getWaitAttrName());
}

::mlir::Attribute ExitDataOp::removeFinalizeAttr() {
  return (*this)->removeAttr(getFinalizeAttrName());
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::UnitAttr finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), wait);
  }
  if (finalize) {
    odsState.addAttribute(getFinalizeAttrName(odsState.name), finalize);
  }
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::UnitAttr finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), wait);
  }
  if (finalize) {
    odsState.addAttribute(getFinalizeAttrName(odsState.name), finalize);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/bool finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), ((wait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (finalize) {
    odsState.addAttribute(getFinalizeAttrName(odsState.name), ((finalize) ? odsBuilder.getUnitAttr() : nullptr));
  }
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/bool finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), ((wait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (finalize) {
    odsState.addAttribute(getFinalizeAttrName(odsState.name), ((finalize) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExitDataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_finalize;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getFinalizeAttrName()) {
      tblgen_finalize = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getWaitAttrName()) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 8)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 8 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_async, "async")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_wait, "wait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_finalize, "finalize")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExitDataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExitDataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deleteOperandsOperands;
  ::llvm::SMLoc deleteOperandsOperandsLoc;
  (void)deleteOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deleteOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> detachOperandsOperands;
  ::llvm::SMLoc detachOperandsOperandsLoc;
  (void)detachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> detachOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dataClauseOperandsOperands;
  ::llvm::SMLoc dataClauseOperandsOperandsLoc;
  (void)dataClauseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> dataClauseOperandsTypes;
  bool ifClause = false;
  bool asyncClause = false;
  bool wait_devnumClause = false;
  bool waitClause = false;
  bool copyoutClause = false;
  bool deleteClause = false;
  bool detachClause = false;
  bool dataOperandsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {

  if (wait_devnumClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait_devnum` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  wait_devnumClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait"))) {

  if (waitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  waitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout"))) {

  if (copyoutClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyoutClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("delete"))) {

  if (deleteClause) {
    return parser.emitError(parser.getNameLoc())
          << "`delete` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deleteClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  deleteOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deleteOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deleteOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("detach"))) {

  if (detachClause) {
    return parser.emitError(parser.getNameLoc())
          << "`detach` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  detachClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  detachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(detachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(detachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("dataOperands"))) {

  if (dataOperandsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`dataOperands` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dataOperandsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  dataClauseOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dataClauseOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(dataClauseOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(deleteOperandsOperands.size()), static_cast<int32_t>(detachOperandsOperands.size()), static_cast<int32_t>(dataClauseOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deleteOperandsOperands, deleteOperandsTypes, deleteOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(detachOperandsOperands, detachOperandsTypes, detachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataClauseOperandsOperands, dataClauseOperandsTypes, dataClauseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExitDataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getAsyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(getAsyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = getWaitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getWaitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(getWaitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitOperands().size()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << getWaitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getWaitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutOperands().size()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDeleteOperands().size()) {
  _odsPrinter << ' ' << "delete";
  _odsPrinter << "(";
  _odsPrinter << getDeleteOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDeleteOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDetachOperands().size()) {
  _odsPrinter << ' ' << "detach";
  _odsPrinter << "(";
  _odsPrinter << getDetachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDetachOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDataClauseOperands().size()) {
  _odsPrinter << ' ' << "dataOperands";
  _odsPrinter << "(";
  _odsPrinter << getDataClauseOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDataClauseOperands().getTypes();
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("async");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("wait");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getFinalizeAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("finalize");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ExitDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::GetDevicePtrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetDevicePtrOpGenericAdaptorBase::GetDevicePtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.getdeviceptr", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetDevicePtrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetDevicePtrOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr GetDevicePtrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr GetDevicePtrOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GetDevicePtrOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause GetDevicePtrOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr GetDevicePtrOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GetDevicePtrOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool GetDevicePtrOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr GetDevicePtrOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GetDevicePtrOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool GetDevicePtrOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr GetDevicePtrOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GetDevicePtrOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > GetDevicePtrOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
GetDevicePtrOpAdaptor::GetDevicePtrOpAdaptor(GetDevicePtrOp op) : GetDevicePtrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetDevicePtrOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.getdeviceptr' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == GetDevicePtrOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GetDevicePtrOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GetDevicePtrOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GetDevicePtrOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GetDevicePtrOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.getdeviceptr' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.getdeviceptr' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.getdeviceptr' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.getdeviceptr' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.getdeviceptr' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetDevicePtrOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range GetDevicePtrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> GetDevicePtrOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> GetDevicePtrOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range GetDevicePtrOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange GetDevicePtrOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GetDevicePtrOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GetDevicePtrOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> GetDevicePtrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetDevicePtrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> GetDevicePtrOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr GetDevicePtrOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause GetDevicePtrOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr GetDevicePtrOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool GetDevicePtrOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr GetDevicePtrOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool GetDevicePtrOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr GetDevicePtrOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > GetDevicePtrOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void GetDevicePtrOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void GetDevicePtrOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void GetDevicePtrOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void GetDevicePtrOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void GetDevicePtrOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void GetDevicePtrOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void GetDevicePtrOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void GetDevicePtrOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute GetDevicePtrOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void GetDevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void GetDevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetDevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void GetDevicePtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetDevicePtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void GetDevicePtrOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_getdeviceptr));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult GetDevicePtrOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetDevicePtrOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GetDevicePtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetDevicePtrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_getdeviceptr)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::GetDevicePtrOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InitOpGenericAdaptorBase::InitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.init", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InitOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr InitOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
InitOpAdaptor::InitOpAdaptor(InitOp op) : InitOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InitOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.init' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == InitOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.init' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> InitOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range InitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InitOp::getDeviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value InitOp::getDeviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> InitOp::getIfCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::MutableOperandRange InitOp::getDeviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InitOp::getDeviceNumOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InitOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> InitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void InitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
}

void InitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InitOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InitOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceNumOperandOperands;
  ::llvm::SMLoc deviceNumOperandOperandsLoc;
  (void)deviceNumOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceNumOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  bool device_typeClause = false;
  bool device_numClause = false;
  bool ifClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("device_type"))) {

  if (device_typeClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device_type` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  device_typeClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("device_num"))) {

  if (device_numClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device_num` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  device_numClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceNumOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(deviceNumOperandOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceNumOperandOperands, deviceNumOperandTypes, deviceNumOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getDeviceTypeOperands().size()) {
  _odsPrinter << ' ' << "device_type";
  _odsPrinter << "(";
  _odsPrinter << getDeviceTypeOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDeviceTypeOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDeviceNumOperand()) {
  _odsPrinter << ' ' << "device_num";
  _odsPrinter << "(";
  if (::mlir::Value value = getDeviceNumOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getDeviceNumOperand() ? ::llvm::ArrayRef<::mlir::Type>(getDeviceNumOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::InitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::KernelsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
KernelsOpGenericAdaptorBase::KernelsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.kernels", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> KernelsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, KernelsOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr KernelsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr KernelsOpGenericAdaptorBase::getAsyncAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, KernelsOp::getAsyncAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool KernelsOpGenericAdaptorBase::getAsyncAttr() {
  auto attr = getAsyncAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr KernelsOpGenericAdaptorBase::getWaitAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, KernelsOp::getWaitAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool KernelsOpGenericAdaptorBase::getWaitAttr() {
  auto attr = getWaitAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr KernelsOpGenericAdaptorBase::getSelfAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, KernelsOp::getSelfAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool KernelsOpGenericAdaptorBase::getSelfAttr() {
  auto attr = getSelfAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ClauseDefaultValueAttr KernelsOpGenericAdaptorBase::getDefaultAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, KernelsOp::getDefaultAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
  return attr;
}

::std::optional<::mlir::acc::ClauseDefaultValue> KernelsOpGenericAdaptorBase::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &KernelsOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange KernelsOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
KernelsOpAdaptor::KernelsOpAdaptor(KernelsOp op) : KernelsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult KernelsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.kernels' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == KernelsOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == KernelsOp::getAsyncAttrAttrName(*odsOpName)) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == KernelsOp::getDefaultAttrAttrName(*odsOpName)) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == KernelsOp::getSelfAttrAttrName(*odsOpName)) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == KernelsOp::getWaitAttrAttrName(*odsOpName)) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 19)
      return emitError(loc, "'acc.kernels' op ""'operand_segment_sizes' attribute for specifying operand segments must have 19 "
                "elements, but got ") << numElements;
  }

  if (tblgen_asyncAttr && !((tblgen_asyncAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.kernels' op ""attribute 'asyncAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_waitAttr && !((tblgen_waitAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.kernels' op ""attribute 'waitAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_selfAttr && !((tblgen_selfAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.kernels' op ""attribute 'selfAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_defaultAttr && !((tblgen_defaultAttr.isa<::mlir::acc::ClauseDefaultValueAttr>())))
    return emitError(loc, "'acc.kernels' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> KernelsOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range KernelsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value KernelsOp::getAsync() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range KernelsOp::getWaitOperands() {
  return getODSOperands(1);
}

::mlir::Value KernelsOp::getNumGangs() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value KernelsOp::getNumWorkers() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value KernelsOp::getVectorLength() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> KernelsOp::getIfCond() {
  auto operands = getODSOperands(5);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> KernelsOp::getSelfCond() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range KernelsOp::getCopyOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range KernelsOp::getCopyinOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range KernelsOp::getCopyinReadonlyOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range KernelsOp::getCopyoutOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range KernelsOp::getCopyoutZeroOperands() {
  return getODSOperands(11);
}

::mlir::Operation::operand_range KernelsOp::getCreateOperands() {
  return getODSOperands(12);
}

::mlir::Operation::operand_range KernelsOp::getCreateZeroOperands() {
  return getODSOperands(13);
}

::mlir::Operation::operand_range KernelsOp::getNoCreateOperands() {
  return getODSOperands(14);
}

::mlir::Operation::operand_range KernelsOp::getPresentOperands() {
  return getODSOperands(15);
}

::mlir::Operation::operand_range KernelsOp::getDevicePtrOperands() {
  return getODSOperands(16);
}

::mlir::Operation::operand_range KernelsOp::getAttachOperands() {
  return getODSOperands(17);
}

::mlir::Operation::operand_range KernelsOp::getDataClauseOperands() {
  return getODSOperands(18);
}

::mlir::MutableOperandRange KernelsOp::getAsyncMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getNumGangsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getNumWorkersMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getVectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getSelfCondMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCopyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCopyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCopyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCopyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCopyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(12);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(12u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getCreateZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(13);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(13u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getNoCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(14);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(14u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getPresentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(15);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(15u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getDevicePtrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(16);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(16u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getAttachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(17);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(17u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange KernelsOp::getDataClauseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(18);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(18u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> KernelsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range KernelsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &KernelsOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr KernelsOp::getAsyncAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool KernelsOp::getAsyncAttr() {
  auto attr = getAsyncAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr KernelsOp::getWaitAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getWaitAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool KernelsOp::getWaitAttr() {
  auto attr = getWaitAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr KernelsOp::getSelfAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSelfAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool KernelsOp::getSelfAttr() {
  auto attr = getSelfAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ClauseDefaultValueAttr KernelsOp::getDefaultAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDefaultAttrAttrName()).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
}

::std::optional<::mlir::acc::ClauseDefaultValue> KernelsOp::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

void KernelsOp::setAsyncAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrAttrName(), attr);
}

void KernelsOp::setAsyncAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrAttrName());
}

void KernelsOp::setWaitAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWaitAttrAttrName(), attr);
}

void KernelsOp::setWaitAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWaitAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWaitAttrAttrName());
}

void KernelsOp::setSelfAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getSelfAttrAttrName(), attr);
}

void KernelsOp::setSelfAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSelfAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getSelfAttrAttrName());
}

void KernelsOp::setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr) {
  (*this)->setAttr(getDefaultAttrAttrName(), attr);
}

void KernelsOp::setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDefaultAttrAttrName(), ::mlir::acc::ClauseDefaultValueAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getDefaultAttrAttrName());
}

::mlir::Attribute KernelsOp::removeAsyncAttrAttr() {
  return (*this)->removeAttr(getAsyncAttrAttrName());
}

::mlir::Attribute KernelsOp::removeWaitAttrAttr() {
  return (*this)->removeAttr(getWaitAttrAttrName());
}

::mlir::Attribute KernelsOp::removeSelfAttrAttr() {
  return (*this)->removeAttr(getSelfAttrAttrName());
}

::mlir::Attribute KernelsOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr(getDefaultAttrAttrName());
}

void KernelsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), selfAttr);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void KernelsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), selfAttr);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void KernelsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), ((asyncAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), ((waitAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), ((selfAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void KernelsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), ((asyncAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), ((waitAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), ((selfAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void KernelsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult KernelsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrAttrName()) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDefaultAttrAttrName()) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSelfAttrAttrName()) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getWaitAttrAttrName()) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 19)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 19 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_asyncAttr, "asyncAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_waitAttr, "waitAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_selfAttr, "selfAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps3(*this, tblgen_defaultAttr, "defaultAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    if (valueGroup5.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup5.size();
    }

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    if (valueGroup6.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup6.size();
    }

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup12 = getODSOperands(12);

    for (auto v : valueGroup12) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup13 = getODSOperands(13);

    for (auto v : valueGroup13) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup14 = getODSOperands(14);

    for (auto v : valueGroup14) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup15 = getODSOperands(15);

    for (auto v : valueGroup15) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup16 = getODSOperands(16);

    for (auto v : valueGroup16) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup17 = getODSOperands(17);

    for (auto v : valueGroup17) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup18 = getODSOperands(18);

    for (auto v : valueGroup18) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult KernelsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult KernelsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dataClauseOperandsOperands;
  ::llvm::SMLoc dataClauseOperandsOperandsLoc;
  (void)dataClauseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> dataClauseOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperands;
  ::llvm::SMLoc asyncOperandsLoc;
  (void)asyncOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyOperandsOperands;
  ::llvm::SMLoc copyOperandsOperandsLoc;
  (void)copyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinReadonlyOperandsOperands;
  ::llvm::SMLoc copyinReadonlyOperandsOperandsLoc;
  (void)copyinReadonlyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinReadonlyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutZeroOperandsOperands;
  ::llvm::SMLoc copyoutZeroOperandsOperandsLoc;
  (void)copyoutZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> devicePtrOperandsOperands;
  ::llvm::SMLoc devicePtrOperandsOperandsLoc;
  (void)devicePtrOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> devicePtrOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> noCreateOperandsOperands;
  ::llvm::SMLoc noCreateOperandsOperandsLoc;
  (void)noCreateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> noCreateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> numGangsOperands;
  ::llvm::SMLoc numGangsOperandsLoc;
  (void)numGangsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> numGangsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> numWorkersOperands;
  ::llvm::SMLoc numWorkersOperandsLoc;
  (void)numWorkersOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> numWorkersTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> presentOperandsOperands;
  ::llvm::SMLoc presentOperandsOperandsLoc;
  (void)presentOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> presentOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> vectorLengthOperands;
  ::llvm::SMLoc vectorLengthOperandsLoc;
  (void)vectorLengthOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> vectorLengthTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> selfCondOperands;
  ::llvm::SMLoc selfCondOperandsLoc;
  (void)selfCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool dataOperandsClause = false;
  bool attachClause = false;
  bool asyncClause = false;
  bool copyClause = false;
  bool copyinClause = false;
  bool copyin_readonlyClause = false;
  bool copyoutClause = false;
  bool copyout_zeroClause = false;
  bool createClause = false;
  bool create_zeroClause = false;
  bool deviceptrClause = false;
  bool no_createClause = false;
  bool num_gangsClause = false;
  bool num_workersClause = false;
  bool presentClause = false;
  bool vector_lengthClause = false;
  bool waitClause = false;
  bool selfClause = false;
  bool ifClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("dataOperands"))) {

  if (dataOperandsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`dataOperands` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dataOperandsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  dataClauseOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dataClauseOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(dataClauseOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("attach"))) {

  if (attachClause) {
    return parser.emitError(parser.getNameLoc())
          << "`attach` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  attachClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copy"))) {

  if (copyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copy` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin"))) {

  if (copyinClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyinClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin_readonly"))) {

  if (copyin_readonlyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin_readonly` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyin_readonlyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinReadonlyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinReadonlyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinReadonlyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout"))) {

  if (copyoutClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyoutClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout_zero"))) {

  if (copyout_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyout_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create"))) {

  if (createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create_zero"))) {

  if (create_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  create_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("deviceptr"))) {

  if (deviceptrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`deviceptr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deviceptrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  devicePtrOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(devicePtrOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(devicePtrOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("no_create"))) {

  if (no_createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`no_create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  no_createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  noCreateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(noCreateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(noCreateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_gangs"))) {

  if (num_gangsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_gangs` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_gangsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    numGangsOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numGangsOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numGangsTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_workers"))) {

  if (num_workersClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_workers` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_workersClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    numWorkersOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numWorkersOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numWorkersTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("present"))) {

  if (presentClause) {
    return parser.emitError(parser.getNameLoc())
          << "`present` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  presentClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  presentOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(presentOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(presentOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("vector_length"))) {

  if (vector_lengthClause) {
    return parser.emitError(parser.getNameLoc())
          << "`vector_length` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  vector_lengthClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    vectorLengthOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      vectorLengthOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      vectorLengthTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait"))) {

  if (waitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  waitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("self"))) {

  if (selfClause) {
    return parser.emitError(parser.getNameLoc())
          << "`self` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  selfClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    selfCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      selfCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(asyncOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(numGangsOperands.size()), static_cast<int32_t>(numWorkersOperands.size()), static_cast<int32_t>(vectorLengthOperands.size()), static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(selfCondOperands.size()), static_cast<int32_t>(copyOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(copyinReadonlyOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(copyoutZeroOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(noCreateOperandsOperands.size()), static_cast<int32_t>(presentOperandsOperands.size()), static_cast<int32_t>(devicePtrOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size()), static_cast<int32_t>(dataClauseOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(asyncOperands, asyncTypes, asyncOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(numGangsOperands, numGangsTypes, numGangsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(numWorkersOperands, numWorkersTypes, numWorkersOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(vectorLengthOperands, vectorLengthTypes, vectorLengthOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(selfCondOperands, odsBuildableType0, selfCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyOperandsOperands, copyOperandsTypes, copyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinReadonlyOperandsOperands, copyinReadonlyOperandsTypes, copyinReadonlyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutZeroOperandsOperands, copyoutZeroOperandsTypes, copyoutZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(noCreateOperandsOperands, noCreateOperandsTypes, noCreateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(presentOperandsOperands, presentOperandsTypes, presentOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(devicePtrOperandsOperands, devicePtrOperandsTypes, devicePtrOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataClauseOperandsOperands, dataClauseOperandsTypes, dataClauseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void KernelsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getDataClauseOperands().size()) {
  _odsPrinter << ' ' << "dataOperands";
  _odsPrinter << "(";
  _odsPrinter << getDataClauseOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDataClauseOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAttachOperands().size()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << getAttachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getAttachOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAsync()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsync())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsync() ? ::llvm::ArrayRef<::mlir::Type>(getAsync().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getCopyOperands().size()) {
  _odsPrinter << ' ' << "copy";
  _odsPrinter << "(";
  _odsPrinter << getCopyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinOperands().size()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << getCopyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinReadonlyOperands().size()) {
  _odsPrinter << ' ' << "copyin_readonly";
  _odsPrinter << "(";
  _odsPrinter << getCopyinReadonlyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinReadonlyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutOperands().size()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutZeroOperands().size()) {
  _odsPrinter << ' ' << "copyout_zero";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateOperands().size()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << getCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateZeroOperands().size()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << getCreateZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDevicePtrOperands().size()) {
  _odsPrinter << ' ' << "deviceptr";
  _odsPrinter << "(";
  _odsPrinter << getDevicePtrOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDevicePtrOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getNoCreateOperands().size()) {
  _odsPrinter << ' ' << "no_create";
  _odsPrinter << "(";
  _odsPrinter << getNoCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getNoCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getNumGangs()) {
  _odsPrinter << ' ' << "num_gangs";
  _odsPrinter << "(";
  if (::mlir::Value value = getNumGangs())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getNumGangs() ? ::llvm::ArrayRef<::mlir::Type>(getNumGangs().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNumWorkers()) {
  _odsPrinter << ' ' << "num_workers";
  _odsPrinter << "(";
  if (::mlir::Value value = getNumWorkers())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getNumWorkers() ? ::llvm::ArrayRef<::mlir::Type>(getNumWorkers().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getPresentOperands().size()) {
  _odsPrinter << ' ' << "present";
  _odsPrinter << "(";
  _odsPrinter << getPresentOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getPresentOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getVectorLength()) {
  _odsPrinter << ' ' << "vector_length";
  _odsPrinter << "(";
  if (::mlir::Value value = getVectorLength())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVectorLength() ? ::llvm::ArrayRef<::mlir::Type>(getVectorLength().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitOperands().size()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << getWaitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getWaitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getSelfCond()) {
  _odsPrinter << ' ' << "self";
  _odsPrinter << "(";
  if (::mlir::Value value = getSelfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("asyncAttr");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWaitAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("waitAttr");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSelfAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("selfAttr");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::KernelsOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoopOpGenericAdaptorBase::LoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.loop", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoopOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr LoopOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopOpGenericAdaptorBase::getCollapseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::getCollapseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> LoopOpGenericAdaptorBase::getCollapse() {
  auto attr = getCollapseAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr LoopOpGenericAdaptorBase::getSeqAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LoopOp::getSeqAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoopOpGenericAdaptorBase::getSeq() {
  auto attr = getSeqAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpGenericAdaptorBase::getIndependentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::getIndependentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoopOpGenericAdaptorBase::getIndependent() {
  auto attr = getIndependentAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpGenericAdaptorBase::getAuto_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::getAuto_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoopOpGenericAdaptorBase::getAuto_() {
  auto attr = getAuto_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpGenericAdaptorBase::getHasGangAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::getHasGangAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoopOpGenericAdaptorBase::getHasGang() {
  auto attr = getHasGangAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpGenericAdaptorBase::getHasWorkerAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::getHasWorkerAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoopOpGenericAdaptorBase::getHasWorker() {
  auto attr = getHasWorkerAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpGenericAdaptorBase::getHasVectorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::getHasVectorAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoopOpGenericAdaptorBase::getHasVector() {
  auto attr = getHasVectorAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr LoopOpGenericAdaptorBase::getReductionOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LoopOp::getReductionOpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
  return attr;
}

::std::optional<::mlir::acc::ReductionOp> LoopOpGenericAdaptorBase::getReductionOp() {
  auto attr = getReductionOpAttr();
  return attr ? ::std::optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &LoopOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange LoopOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
LoopOpAdaptor::LoopOpAdaptor(LoopOp op) : LoopOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_auto_;
  ::mlir::Attribute tblgen_collapse;
  ::mlir::Attribute tblgen_hasGang;
  ::mlir::Attribute tblgen_hasVector;
  ::mlir::Attribute tblgen_hasWorker;
  ::mlir::Attribute tblgen_independent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.loop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == LoopOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == LoopOp::getAuto_AttrName(*odsOpName)) {
      tblgen_auto_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::getCollapseAttrName(*odsOpName)) {
      tblgen_collapse = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::getHasGangAttrName(*odsOpName)) {
      tblgen_hasGang = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::getHasVectorAttrName(*odsOpName)) {
      tblgen_hasVector = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::getHasWorkerAttrName(*odsOpName)) {
      tblgen_hasWorker = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::getIndependentAttrName(*odsOpName)) {
      tblgen_independent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_seq;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoopOp::getReductionOpAttrName(*odsOpName)) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::getSeqAttrName(*odsOpName)) {
      tblgen_seq = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'acc.loop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_collapse && !(((tblgen_collapse.isa<::mlir::IntegerAttr>())) && ((tblgen_collapse.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'acc.loop' op ""attribute 'collapse' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_seq && !((tblgen_seq.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'seq' failed to satisfy constraint: unit attribute");

  if (tblgen_independent && !((tblgen_independent.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'independent' failed to satisfy constraint: unit attribute");

  if (tblgen_auto_ && !((tblgen_auto_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'auto_' failed to satisfy constraint: unit attribute");

  if (tblgen_hasGang && !((tblgen_hasGang.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'hasGang' failed to satisfy constraint: unit attribute");

  if (tblgen_hasWorker && !((tblgen_hasWorker.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'hasWorker' failed to satisfy constraint: unit attribute");

  if (tblgen_hasVector && !((tblgen_hasVector.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'hasVector' failed to satisfy constraint: unit attribute");

  if (tblgen_reductionOp && !((tblgen_reductionOp.isa<::mlir::acc::ReductionOpAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range LoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopOp::getGangNum() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value LoopOp::getGangStatic() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value LoopOp::getWorkerNum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value LoopOp::getVectorLength() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range LoopOp::getTileOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range LoopOp::getPrivateOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range LoopOp::getReductionOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange LoopOp::getGangNumMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::getGangStaticMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::getWorkerNumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::getVectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::getTileOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::getPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::getReductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range LoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range LoopOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &LoopOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr LoopOp::getCollapseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCollapseAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> LoopOp::getCollapse() {
  auto attr = getCollapseAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr LoopOp::getSeqAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSeqAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::getSeq() {
  auto attr = getSeqAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::getIndependentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIndependentAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::getIndependent() {
  auto attr = getIndependentAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::getAuto_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAuto_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::getAuto_() {
  auto attr = getAuto_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::getHasGangAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getHasGangAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::getHasGang() {
  auto attr = getHasGangAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::getHasWorkerAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getHasWorkerAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::getHasWorker() {
  auto attr = getHasWorkerAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::getHasVectorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getHasVectorAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::getHasVector() {
  auto attr = getHasVectorAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr LoopOp::getReductionOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionOpAttrName()).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
}

::std::optional<::mlir::acc::ReductionOp> LoopOp::getReductionOp() {
  auto attr = getReductionOpAttr();
  return attr ? ::std::optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::std::nullopt);
}

void LoopOp::setCollapseAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCollapseAttrName(), attr);
}

void LoopOp::setCollapse(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCollapseAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getCollapseAttrName());
}

void LoopOp::setSeqAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getSeqAttrName(), attr);
}

void LoopOp::setSeq(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSeqAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getSeqAttrName());
}

void LoopOp::setIndependentAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIndependentAttrName(), attr);
}

void LoopOp::setIndependent(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIndependentAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getIndependentAttrName());
}

void LoopOp::setAuto_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAuto_AttrName(), attr);
}

void LoopOp::setAuto_(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAuto_AttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAuto_AttrName());
}

void LoopOp::setHasGangAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getHasGangAttrName(), attr);
}

void LoopOp::setHasGang(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHasGangAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getHasGangAttrName());
}

void LoopOp::setHasWorkerAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getHasWorkerAttrName(), attr);
}

void LoopOp::setHasWorker(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHasWorkerAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getHasWorkerAttrName());
}

void LoopOp::setHasVectorAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getHasVectorAttrName(), attr);
}

void LoopOp::setHasVector(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHasVectorAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getHasVectorAttrName());
}

void LoopOp::setReductionOpAttr(::mlir::acc::ReductionOpAttr attr) {
  (*this)->setAttr(getReductionOpAttrName(), attr);
}

void LoopOp::setReductionOp(::std::optional<::mlir::acc::ReductionOp> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getReductionOpAttrName(), ::mlir::acc::ReductionOpAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getReductionOpAttrName());
}

::mlir::Attribute LoopOp::removeCollapseAttr() {
  return (*this)->removeAttr(getCollapseAttrName());
}

::mlir::Attribute LoopOp::removeSeqAttr() {
  return (*this)->removeAttr(getSeqAttrName());
}

::mlir::Attribute LoopOp::removeIndependentAttr() {
  return (*this)->removeAttr(getIndependentAttrName());
}

::mlir::Attribute LoopOp::removeAuto_Attr() {
  return (*this)->removeAttr(getAuto_AttrName());
}

::mlir::Attribute LoopOp::removeHasGangAttr() {
  return (*this)->removeAttr(getHasGangAttrName());
}

::mlir::Attribute LoopOp::removeHasWorkerAttr() {
  return (*this)->removeAttr(getHasWorkerAttrName());
}

::mlir::Attribute LoopOp::removeHasVectorAttr() {
  return (*this)->removeAttr(getHasVectorAttrName());
}

::mlir::Attribute LoopOp::removeReductionOpAttr() {
  return (*this)->removeAttr(getReductionOpAttrName());
}

void LoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, /*optional*/::mlir::UnitAttr hasGang, /*optional*/::mlir::UnitAttr hasWorker, /*optional*/::mlir::UnitAttr hasVector, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands) {
  if (gangNum)
    odsState.addOperands(gangNum);
  if (gangStatic)
    odsState.addOperands(gangStatic);
  if (workerNum)
    odsState.addOperands(workerNum);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  odsState.addOperands(tileOperands);
  odsState.addOperands(privateOperands);
  odsState.addOperands(reductionOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(gangNum ? 1 : 0), (gangStatic ? 1 : 0), (workerNum ? 1 : 0), (vectorLength ? 1 : 0), static_cast<int32_t>(tileOperands.size()), static_cast<int32_t>(privateOperands.size()), static_cast<int32_t>(reductionOperands.size())}));
  if (collapse) {
    odsState.addAttribute(getCollapseAttrName(odsState.name), collapse);
  }
  if (seq) {
    odsState.addAttribute(getSeqAttrName(odsState.name), seq);
  }
  if (independent) {
    odsState.addAttribute(getIndependentAttrName(odsState.name), independent);
  }
  if (auto_) {
    odsState.addAttribute(getAuto_AttrName(odsState.name), auto_);
  }
  if (hasGang) {
    odsState.addAttribute(getHasGangAttrName(odsState.name), hasGang);
  }
  if (hasWorker) {
    odsState.addAttribute(getHasWorkerAttrName(odsState.name), hasWorker);
  }
  if (hasVector) {
    odsState.addAttribute(getHasVectorAttrName(odsState.name), hasVector);
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, /*optional*/bool hasGang, /*optional*/bool hasWorker, /*optional*/bool hasVector, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands) {
  if (gangNum)
    odsState.addOperands(gangNum);
  if (gangStatic)
    odsState.addOperands(gangStatic);
  if (workerNum)
    odsState.addOperands(workerNum);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  odsState.addOperands(tileOperands);
  odsState.addOperands(privateOperands);
  odsState.addOperands(reductionOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(gangNum ? 1 : 0), (gangStatic ? 1 : 0), (workerNum ? 1 : 0), (vectorLength ? 1 : 0), static_cast<int32_t>(tileOperands.size()), static_cast<int32_t>(privateOperands.size()), static_cast<int32_t>(reductionOperands.size())}));
  if (collapse) {
    odsState.addAttribute(getCollapseAttrName(odsState.name), collapse);
  }
  if (seq) {
    odsState.addAttribute(getSeqAttrName(odsState.name), ((seq) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (independent) {
    odsState.addAttribute(getIndependentAttrName(odsState.name), ((independent) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (auto_) {
    odsState.addAttribute(getAuto_AttrName(odsState.name), ((auto_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (hasGang) {
    odsState.addAttribute(getHasGangAttrName(odsState.name), ((hasGang) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (hasWorker) {
    odsState.addAttribute(getHasWorkerAttrName(odsState.name), ((hasWorker) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (hasVector) {
    odsState.addAttribute(getHasVectorAttrName(odsState.name), ((hasVector) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_auto_;
  ::mlir::Attribute tblgen_collapse;
  ::mlir::Attribute tblgen_hasGang;
  ::mlir::Attribute tblgen_hasVector;
  ::mlir::Attribute tblgen_hasWorker;
  ::mlir::Attribute tblgen_independent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAuto_AttrName()) {
      tblgen_auto_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCollapseAttrName()) {
      tblgen_collapse = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getHasGangAttrName()) {
      tblgen_hasGang = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getHasVectorAttrName()) {
      tblgen_hasVector = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getHasWorkerAttrName()) {
      tblgen_hasWorker = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIndependentAttrName()) {
      tblgen_independent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_seq;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReductionOpAttrName()) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSeqAttrName()) {
      tblgen_seq = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps5(*this, tblgen_collapse, "collapse")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_seq, "seq")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_independent, "independent")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_auto_, "auto_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_hasGang, "hasGang")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_hasWorker, "hasWorker")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_hasVector, "hasVector")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps6(*this, tblgen_reductionOp, "reductionOp")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult LoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> gangNumOperands;
  ::llvm::SMLoc gangNumOperandsLoc;
  (void)gangNumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> gangNumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> gangStaticOperands;
  ::llvm::SMLoc gangStaticOperandsLoc;
  (void)gangStaticOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> gangStaticTypes;
  ::mlir::UnitAttr hasGangAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> workerNumOperands;
  ::llvm::SMLoc workerNumOperandsLoc;
  (void)workerNumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> workerNumTypes;
  ::mlir::UnitAttr hasWorkerAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> vectorLengthOperands;
  ::llvm::SMLoc vectorLengthOperandsLoc;
  (void)vectorLengthOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> vectorLengthTypes;
  ::mlir::UnitAttr hasVectorAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> privateOperandsOperands;
  ::llvm::SMLoc privateOperandsOperandsLoc;
  (void)privateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> privateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> tileOperandsOperands;
  ::llvm::SMLoc tileOperandsOperandsLoc;
  (void)tileOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> tileOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reductionOperandsOperands;
  ::llvm::SMLoc reductionOperandsOperandsLoc;
  (void)reductionOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reductionOperandsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  bool gangClause = false;
  bool workerClause = false;
  bool vectorClause = false;
  bool privateClause = false;
  bool tileClause = false;
  bool reductionClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("gang"))) {

  if (gangClause) {
    return parser.emitError(parser.getNameLoc())
          << "`gang` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  gangClause = true;
  {
    gangNumOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> gangNumOperand;
    ::mlir::Type gangNumType;
    gangStaticOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> gangStaticOperand;
    ::mlir::Type gangStaticType;
    if (parseGangClause(parser, gangNumOperand, gangNumType, gangStaticOperand, gangStaticType, hasGangAttr))
      return ::mlir::failure();
    if (gangNumOperand.has_value())
      gangNumOperands.push_back(*gangNumOperand);
    if (gangNumType)
      gangNumTypes.push_back(gangNumType);
    if (gangStaticOperand.has_value())
      gangStaticOperands.push_back(*gangStaticOperand);
    if (gangStaticType)
      gangStaticTypes.push_back(gangStaticType);
    if (hasGangAttr)
      result.addAttribute("hasGang", hasGangAttr);
  }
    } else if (succeeded(parser.parseOptionalKeyword("worker"))) {

  if (workerClause) {
    return parser.emitError(parser.getNameLoc())
          << "`worker` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  workerClause = true;
  {
    workerNumOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> workerNumOperand;
    ::mlir::Type workerNumType;
    if (parseWorkerClause(parser, workerNumOperand, workerNumType, hasWorkerAttr))
      return ::mlir::failure();
    if (workerNumOperand.has_value())
      workerNumOperands.push_back(*workerNumOperand);
    if (workerNumType)
      workerNumTypes.push_back(workerNumType);
    if (hasWorkerAttr)
      result.addAttribute("hasWorker", hasWorkerAttr);
  }
    } else if (succeeded(parser.parseOptionalKeyword("vector"))) {

  if (vectorClause) {
    return parser.emitError(parser.getNameLoc())
          << "`vector` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  vectorClause = true;
  {
    vectorLengthOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> vectorLengthOperand;
    ::mlir::Type vectorLengthType;
    if (parseVectorClause(parser, vectorLengthOperand, vectorLengthType, hasVectorAttr))
      return ::mlir::failure();
    if (vectorLengthOperand.has_value())
      vectorLengthOperands.push_back(*vectorLengthOperand);
    if (vectorLengthType)
      vectorLengthTypes.push_back(vectorLengthType);
    if (hasVectorAttr)
      result.addAttribute("hasVector", hasVectorAttr);
  }
    } else if (succeeded(parser.parseOptionalKeyword("private"))) {

  if (privateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`private` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  privateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  privateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(privateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(privateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("tile"))) {

  if (tileClause) {
    return parser.emitError(parser.getNameLoc())
          << "`tile` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  tileClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  tileOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(tileOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(tileOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  reductionOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(reductionOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(reductionOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(gangNumOperands.size()), static_cast<int32_t>(gangStaticOperands.size()), static_cast<int32_t>(workerNumOperands.size()), static_cast<int32_t>(vectorLengthOperands.size()), static_cast<int32_t>(tileOperandsOperands.size()), static_cast<int32_t>(privateOperandsOperands.size()), static_cast<int32_t>(reductionOperandsOperands.size())}));
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(gangNumOperands, gangNumTypes, gangNumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gangStaticOperands, gangStaticTypes, gangStaticOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(workerNumOperands, workerNumTypes, workerNumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(vectorLengthOperands, vectorLengthTypes, vectorLengthOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(tileOperandsOperands, tileOperandsTypes, tileOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(privateOperandsOperands, privateOperandsTypes, privateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reductionOperandsOperands, reductionOperandsTypes, reductionOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getGangNum() || getGangStatic() || getHasGangAttr()) {
  _odsPrinter << ' ' << "gang";
  printGangClause(_odsPrinter, *this, getGangNum(), (getGangNum() ? getGangNum().getType() : ::mlir::Type()), getGangStatic(), (getGangStatic() ? getGangStatic().getType() : ::mlir::Type()), getHasGangAttr());
  }
  if (false || getWorkerNum() || getHasWorkerAttr()) {
  _odsPrinter << ' ' << "worker";
  printWorkerClause(_odsPrinter, *this, getWorkerNum(), (getWorkerNum() ? getWorkerNum().getType() : ::mlir::Type()), getHasWorkerAttr());
  }
  if (false || getVectorLength() || getHasVectorAttr()) {
  _odsPrinter << ' ' << "vector";
  printVectorClause(_odsPrinter, *this, getVectorLength(), (getVectorLength() ? getVectorLength().getType() : ::mlir::Type()), getHasVectorAttr());
  }
  if (false || getPrivateOperands().size()) {
  _odsPrinter << ' ' << "private";
  _odsPrinter << "(";
  _odsPrinter << getPrivateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getPrivateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getTileOperands().size()) {
  _odsPrinter << ' ' << "tile";
  _odsPrinter << "(";
  _odsPrinter << getTileOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getTileOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getReductionOperands().size()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  _odsPrinter << getReductionOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getReductionOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  if (!getResults().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getResults().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("hasGang");
  elidedAttrs.push_back("hasWorker");
  elidedAttrs.push_back("hasVector");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSeqAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("seq");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIndependentAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("independent");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAuto_Attr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("auto_");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHasGangAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("hasGang");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHasWorkerAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("hasWorker");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHasVectorAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("hasVector");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::LoopOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::NoCreateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
NoCreateOpGenericAdaptorBase::NoCreateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.nocreate", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> NoCreateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, NoCreateOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr NoCreateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr NoCreateOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, NoCreateOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause NoCreateOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr NoCreateOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, NoCreateOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool NoCreateOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr NoCreateOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, NoCreateOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool NoCreateOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr NoCreateOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, NoCreateOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > NoCreateOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
NoCreateOpAdaptor::NoCreateOpAdaptor(NoCreateOp op) : NoCreateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult NoCreateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.nocreate' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == NoCreateOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == NoCreateOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == NoCreateOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == NoCreateOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == NoCreateOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.nocreate' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.nocreate' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.nocreate' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.nocreate' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.nocreate' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NoCreateOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range NoCreateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> NoCreateOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> NoCreateOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range NoCreateOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange NoCreateOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange NoCreateOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange NoCreateOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> NoCreateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NoCreateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> NoCreateOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr NoCreateOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause NoCreateOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr NoCreateOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool NoCreateOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr NoCreateOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool NoCreateOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr NoCreateOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > NoCreateOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void NoCreateOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void NoCreateOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void NoCreateOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void NoCreateOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void NoCreateOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void NoCreateOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void NoCreateOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void NoCreateOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute NoCreateOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void NoCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void NoCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NoCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void NoCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NoCreateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void NoCreateOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_no_create));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult NoCreateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NoCreateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult NoCreateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NoCreateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_no_create)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::NoCreateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ParallelOpGenericAdaptorBase::ParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.parallel", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ParallelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ParallelOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ParallelOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ParallelOpGenericAdaptorBase::getAsyncAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ParallelOp::getAsyncAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ParallelOpGenericAdaptorBase::getAsyncAttr() {
  auto attr = getAsyncAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOpGenericAdaptorBase::getWaitAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::getWaitAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ParallelOpGenericAdaptorBase::getWaitAttr() {
  auto attr = getWaitAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOpGenericAdaptorBase::getSelfAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::getSelfAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ParallelOpGenericAdaptorBase::getSelfAttr() {
  auto attr = getSelfAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr ParallelOpGenericAdaptorBase::getReductionOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::getReductionOpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
  return attr;
}

::std::optional<::mlir::acc::ReductionOp> ParallelOpGenericAdaptorBase::getReductionOp() {
  auto attr = getReductionOpAttr();
  return attr ? ::std::optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::std::nullopt);
}

::mlir::acc::ClauseDefaultValueAttr ParallelOpGenericAdaptorBase::getDefaultAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ParallelOp::getDefaultAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
  return attr;
}

::std::optional<::mlir::acc::ClauseDefaultValue> ParallelOpGenericAdaptorBase::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &ParallelOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ParallelOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ParallelOpAdaptor::ParallelOpAdaptor(ParallelOp op) : ParallelOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ParallelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.parallel' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ParallelOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ParallelOp::getAsyncAttrAttrName(*odsOpName)) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::getDefaultAttrAttrName(*odsOpName)) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ParallelOp::getReductionOpAttrName(*odsOpName)) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::getSelfAttrAttrName(*odsOpName)) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::getWaitAttrAttrName(*odsOpName)) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 22)
      return emitError(loc, "'acc.parallel' op ""'operand_segment_sizes' attribute for specifying operand segments must have 22 "
                "elements, but got ") << numElements;
  }

  if (tblgen_asyncAttr && !((tblgen_asyncAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'asyncAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_waitAttr && !((tblgen_waitAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'waitAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_selfAttr && !((tblgen_selfAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'selfAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_reductionOp && !((tblgen_reductionOp.isa<::mlir::acc::ReductionOpAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");

  if (tblgen_defaultAttr && !((tblgen_defaultAttr.isa<::mlir::acc::ClauseDefaultValueAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ParallelOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ParallelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOp::getAsync() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range ParallelOp::getWaitOperands() {
  return getODSOperands(1);
}

::mlir::Value ParallelOp::getNumGangs() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value ParallelOp::getNumWorkers() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value ParallelOp::getVectorLength() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> ParallelOp::getIfCond() {
  auto operands = getODSOperands(5);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> ParallelOp::getSelfCond() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range ParallelOp::getReductionOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range ParallelOp::getCopyOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range ParallelOp::getCopyinOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range ParallelOp::getCopyinReadonlyOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range ParallelOp::getCopyoutOperands() {
  return getODSOperands(11);
}

::mlir::Operation::operand_range ParallelOp::getCopyoutZeroOperands() {
  return getODSOperands(12);
}

::mlir::Operation::operand_range ParallelOp::getCreateOperands() {
  return getODSOperands(13);
}

::mlir::Operation::operand_range ParallelOp::getCreateZeroOperands() {
  return getODSOperands(14);
}

::mlir::Operation::operand_range ParallelOp::getNoCreateOperands() {
  return getODSOperands(15);
}

::mlir::Operation::operand_range ParallelOp::getPresentOperands() {
  return getODSOperands(16);
}

::mlir::Operation::operand_range ParallelOp::getDevicePtrOperands() {
  return getODSOperands(17);
}

::mlir::Operation::operand_range ParallelOp::getAttachOperands() {
  return getODSOperands(18);
}

::mlir::Operation::operand_range ParallelOp::getGangPrivateOperands() {
  return getODSOperands(19);
}

::mlir::Operation::operand_range ParallelOp::getGangFirstPrivateOperands() {
  return getODSOperands(20);
}

::mlir::Operation::operand_range ParallelOp::getDataClauseOperands() {
  return getODSOperands(21);
}

::mlir::MutableOperandRange ParallelOp::getAsyncMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getNumGangsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getNumWorkersMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getVectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getSelfCondMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getReductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCopyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCopyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCopyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCopyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCopyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(12);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(12u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(13);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(13u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getCreateZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(14);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(14u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getNoCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(15);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(15u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getPresentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(16);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(16u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getDevicePtrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(17);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(17u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getAttachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(18);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(18u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getGangPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(19);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(19u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getGangFirstPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(20);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(20u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getDataClauseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(21);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(21u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ParallelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ParallelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ParallelOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr ParallelOp::getAsyncAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::getAsyncAttr() {
  auto attr = getAsyncAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOp::getWaitAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getWaitAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::getWaitAttr() {
  auto attr = getWaitAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOp::getSelfAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSelfAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::getSelfAttr() {
  auto attr = getSelfAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr ParallelOp::getReductionOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionOpAttrName()).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
}

::std::optional<::mlir::acc::ReductionOp> ParallelOp::getReductionOp() {
  auto attr = getReductionOpAttr();
  return attr ? ::std::optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::std::nullopt);
}

::mlir::acc::ClauseDefaultValueAttr ParallelOp::getDefaultAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDefaultAttrAttrName()).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
}

::std::optional<::mlir::acc::ClauseDefaultValue> ParallelOp::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

void ParallelOp::setAsyncAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrAttrName(), attr);
}

void ParallelOp::setAsyncAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrAttrName());
}

void ParallelOp::setWaitAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWaitAttrAttrName(), attr);
}

void ParallelOp::setWaitAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWaitAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWaitAttrAttrName());
}

void ParallelOp::setSelfAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getSelfAttrAttrName(), attr);
}

void ParallelOp::setSelfAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSelfAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getSelfAttrAttrName());
}

void ParallelOp::setReductionOpAttr(::mlir::acc::ReductionOpAttr attr) {
  (*this)->setAttr(getReductionOpAttrName(), attr);
}

void ParallelOp::setReductionOp(::std::optional<::mlir::acc::ReductionOp> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getReductionOpAttrName(), ::mlir::acc::ReductionOpAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getReductionOpAttrName());
}

void ParallelOp::setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr) {
  (*this)->setAttr(getDefaultAttrAttrName(), attr);
}

void ParallelOp::setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDefaultAttrAttrName(), ::mlir::acc::ClauseDefaultValueAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getDefaultAttrAttrName());
}

::mlir::Attribute ParallelOp::removeAsyncAttrAttr() {
  return (*this)->removeAttr(getAsyncAttrAttrName());
}

::mlir::Attribute ParallelOp::removeWaitAttrAttr() {
  return (*this)->removeAttr(getWaitAttrAttrName());
}

::mlir::Attribute ParallelOp::removeSelfAttrAttr() {
  return (*this)->removeAttr(getSelfAttrAttrName());
}

::mlir::Attribute ParallelOp::removeReductionOpAttr() {
  return (*this)->removeAttr(getReductionOpAttrName());
}

::mlir::Attribute ParallelOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr(getDefaultAttrAttrName());
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), selfAttr);
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), selfAttr);
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), ((asyncAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), ((waitAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), ((selfAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), ((asyncAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), ((waitAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), ((selfAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ParallelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrAttrName()) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDefaultAttrAttrName()) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReductionOpAttrName()) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSelfAttrAttrName()) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getWaitAttrAttrName()) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 22)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 22 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_asyncAttr, "asyncAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_waitAttr, "waitAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_selfAttr, "selfAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps6(*this, tblgen_reductionOp, "reductionOp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps3(*this, tblgen_defaultAttr, "defaultAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    if (valueGroup5.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup5.size();
    }

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    if (valueGroup6.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup6.size();
    }

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup12 = getODSOperands(12);

    for (auto v : valueGroup12) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup13 = getODSOperands(13);

    for (auto v : valueGroup13) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup14 = getODSOperands(14);

    for (auto v : valueGroup14) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup15 = getODSOperands(15);

    for (auto v : valueGroup15) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup16 = getODSOperands(16);

    for (auto v : valueGroup16) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup17 = getODSOperands(17);

    for (auto v : valueGroup17) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup18 = getODSOperands(18);

    for (auto v : valueGroup18) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup19 = getODSOperands(19);

    for (auto v : valueGroup19) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup20 = getODSOperands(20);

    for (auto v : valueGroup20) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup21 = getODSOperands(21);

    for (auto v : valueGroup21) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ParallelOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ParallelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dataClauseOperandsOperands;
  ::llvm::SMLoc dataClauseOperandsOperandsLoc;
  (void)dataClauseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> dataClauseOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperands;
  ::llvm::SMLoc asyncOperandsLoc;
  (void)asyncOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyOperandsOperands;
  ::llvm::SMLoc copyOperandsOperandsLoc;
  (void)copyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinReadonlyOperandsOperands;
  ::llvm::SMLoc copyinReadonlyOperandsOperandsLoc;
  (void)copyinReadonlyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinReadonlyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutZeroOperandsOperands;
  ::llvm::SMLoc copyoutZeroOperandsOperandsLoc;
  (void)copyoutZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> devicePtrOperandsOperands;
  ::llvm::SMLoc devicePtrOperandsOperandsLoc;
  (void)devicePtrOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> devicePtrOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> gangFirstPrivateOperandsOperands;
  ::llvm::SMLoc gangFirstPrivateOperandsOperandsLoc;
  (void)gangFirstPrivateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> gangFirstPrivateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> noCreateOperandsOperands;
  ::llvm::SMLoc noCreateOperandsOperandsLoc;
  (void)noCreateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> noCreateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> numGangsOperands;
  ::llvm::SMLoc numGangsOperandsLoc;
  (void)numGangsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> numGangsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> numWorkersOperands;
  ::llvm::SMLoc numWorkersOperandsLoc;
  (void)numWorkersOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> numWorkersTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> gangPrivateOperandsOperands;
  ::llvm::SMLoc gangPrivateOperandsOperandsLoc;
  (void)gangPrivateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> gangPrivateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> presentOperandsOperands;
  ::llvm::SMLoc presentOperandsOperandsLoc;
  (void)presentOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> presentOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> vectorLengthOperands;
  ::llvm::SMLoc vectorLengthOperandsLoc;
  (void)vectorLengthOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> vectorLengthTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> selfCondOperands;
  ::llvm::SMLoc selfCondOperandsLoc;
  (void)selfCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reductionOperandsOperands;
  ::llvm::SMLoc reductionOperandsOperandsLoc;
  (void)reductionOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reductionOperandsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool dataOperandsClause = false;
  bool attachClause = false;
  bool asyncClause = false;
  bool copyClause = false;
  bool copyinClause = false;
  bool copyin_readonlyClause = false;
  bool copyoutClause = false;
  bool copyout_zeroClause = false;
  bool createClause = false;
  bool create_zeroClause = false;
  bool deviceptrClause = false;
  bool firstprivateClause = false;
  bool no_createClause = false;
  bool num_gangsClause = false;
  bool num_workersClause = false;
  bool privateClause = false;
  bool presentClause = false;
  bool vector_lengthClause = false;
  bool waitClause = false;
  bool selfClause = false;
  bool ifClause = false;
  bool reductionClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("dataOperands"))) {

  if (dataOperandsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`dataOperands` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dataOperandsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  dataClauseOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dataClauseOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(dataClauseOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("attach"))) {

  if (attachClause) {
    return parser.emitError(parser.getNameLoc())
          << "`attach` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  attachClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copy"))) {

  if (copyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copy` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin"))) {

  if (copyinClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyinClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin_readonly"))) {

  if (copyin_readonlyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin_readonly` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyin_readonlyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinReadonlyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinReadonlyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinReadonlyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout"))) {

  if (copyoutClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyoutClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout_zero"))) {

  if (copyout_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyout_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create"))) {

  if (createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create_zero"))) {

  if (create_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  create_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("deviceptr"))) {

  if (deviceptrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`deviceptr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deviceptrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  devicePtrOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(devicePtrOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(devicePtrOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("firstprivate"))) {

  if (firstprivateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`firstprivate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  firstprivateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  gangFirstPrivateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(gangFirstPrivateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(gangFirstPrivateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("no_create"))) {

  if (no_createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`no_create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  no_createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  noCreateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(noCreateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(noCreateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_gangs"))) {

  if (num_gangsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_gangs` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_gangsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    numGangsOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numGangsOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numGangsTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_workers"))) {

  if (num_workersClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_workers` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_workersClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    numWorkersOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numWorkersOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      numWorkersTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("private"))) {

  if (privateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`private` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  privateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  gangPrivateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(gangPrivateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(gangPrivateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("present"))) {

  if (presentClause) {
    return parser.emitError(parser.getNameLoc())
          << "`present` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  presentClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  presentOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(presentOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(presentOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("vector_length"))) {

  if (vector_lengthClause) {
    return parser.emitError(parser.getNameLoc())
          << "`vector_length` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  vector_lengthClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    vectorLengthOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      vectorLengthOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      vectorLengthTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait"))) {

  if (waitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  waitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("self"))) {

  if (selfClause) {
    return parser.emitError(parser.getNameLoc())
          << "`self` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  selfClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    selfCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      selfCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  reductionOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(reductionOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(reductionOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(asyncOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(numGangsOperands.size()), static_cast<int32_t>(numWorkersOperands.size()), static_cast<int32_t>(vectorLengthOperands.size()), static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(selfCondOperands.size()), static_cast<int32_t>(reductionOperandsOperands.size()), static_cast<int32_t>(copyOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(copyinReadonlyOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(copyoutZeroOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(noCreateOperandsOperands.size()), static_cast<int32_t>(presentOperandsOperands.size()), static_cast<int32_t>(devicePtrOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size()), static_cast<int32_t>(gangPrivateOperandsOperands.size()), static_cast<int32_t>(gangFirstPrivateOperandsOperands.size()), static_cast<int32_t>(dataClauseOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(asyncOperands, asyncTypes, asyncOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(numGangsOperands, numGangsTypes, numGangsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(numWorkersOperands, numWorkersTypes, numWorkersOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(vectorLengthOperands, vectorLengthTypes, vectorLengthOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(selfCondOperands, odsBuildableType0, selfCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reductionOperandsOperands, reductionOperandsTypes, reductionOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyOperandsOperands, copyOperandsTypes, copyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinReadonlyOperandsOperands, copyinReadonlyOperandsTypes, copyinReadonlyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutZeroOperandsOperands, copyoutZeroOperandsTypes, copyoutZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(noCreateOperandsOperands, noCreateOperandsTypes, noCreateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(presentOperandsOperands, presentOperandsTypes, presentOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(devicePtrOperandsOperands, devicePtrOperandsTypes, devicePtrOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gangPrivateOperandsOperands, gangPrivateOperandsTypes, gangPrivateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gangFirstPrivateOperandsOperands, gangFirstPrivateOperandsTypes, gangFirstPrivateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataClauseOperandsOperands, dataClauseOperandsTypes, dataClauseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ParallelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getDataClauseOperands().size()) {
  _odsPrinter << ' ' << "dataOperands";
  _odsPrinter << "(";
  _odsPrinter << getDataClauseOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDataClauseOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAttachOperands().size()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << getAttachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getAttachOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAsync()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsync())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsync() ? ::llvm::ArrayRef<::mlir::Type>(getAsync().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getCopyOperands().size()) {
  _odsPrinter << ' ' << "copy";
  _odsPrinter << "(";
  _odsPrinter << getCopyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinOperands().size()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << getCopyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinReadonlyOperands().size()) {
  _odsPrinter << ' ' << "copyin_readonly";
  _odsPrinter << "(";
  _odsPrinter << getCopyinReadonlyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinReadonlyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutOperands().size()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutZeroOperands().size()) {
  _odsPrinter << ' ' << "copyout_zero";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateOperands().size()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << getCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateZeroOperands().size()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << getCreateZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDevicePtrOperands().size()) {
  _odsPrinter << ' ' << "deviceptr";
  _odsPrinter << "(";
  _odsPrinter << getDevicePtrOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDevicePtrOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getGangFirstPrivateOperands().size()) {
  _odsPrinter << ' ' << "firstprivate";
  _odsPrinter << "(";
  _odsPrinter << getGangFirstPrivateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getGangFirstPrivateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getNoCreateOperands().size()) {
  _odsPrinter << ' ' << "no_create";
  _odsPrinter << "(";
  _odsPrinter << getNoCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getNoCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getNumGangs()) {
  _odsPrinter << ' ' << "num_gangs";
  _odsPrinter << "(";
  if (::mlir::Value value = getNumGangs())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getNumGangs() ? ::llvm::ArrayRef<::mlir::Type>(getNumGangs().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNumWorkers()) {
  _odsPrinter << ' ' << "num_workers";
  _odsPrinter << "(";
  if (::mlir::Value value = getNumWorkers())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getNumWorkers() ? ::llvm::ArrayRef<::mlir::Type>(getNumWorkers().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getGangPrivateOperands().size()) {
  _odsPrinter << ' ' << "private";
  _odsPrinter << "(";
  _odsPrinter << getGangPrivateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getGangPrivateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getPresentOperands().size()) {
  _odsPrinter << ' ' << "present";
  _odsPrinter << "(";
  _odsPrinter << getPresentOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getPresentOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getVectorLength()) {
  _odsPrinter << ' ' << "vector_length";
  _odsPrinter << "(";
  if (::mlir::Value value = getVectorLength())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVectorLength() ? ::llvm::ArrayRef<::mlir::Type>(getVectorLength().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitOperands().size()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << getWaitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getWaitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getSelfCond()) {
  _odsPrinter << ' ' << "self";
  _odsPrinter << "(";
  if (::mlir::Value value = getSelfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getReductionOperands().size()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  _odsPrinter << getReductionOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getReductionOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("asyncAttr");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWaitAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("waitAttr");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSelfAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("selfAttr");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ParallelOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::PresentOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PresentOpGenericAdaptorBase::PresentOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.present", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PresentOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PresentOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PresentOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::acc::DataClauseAttr PresentOpGenericAdaptorBase::getDataClauseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PresentOp::getDataClauseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
  return attr;
}

::mlir::acc::DataClause PresentOpGenericAdaptorBase::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr PresentOpGenericAdaptorBase::getStructuredAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PresentOp::getStructuredAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool PresentOpGenericAdaptorBase::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr PresentOpGenericAdaptorBase::getImplicitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PresentOp::getImplicitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool PresentOpGenericAdaptorBase::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr PresentOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PresentOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > PresentOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
PresentOpAdaptor::PresentOpAdaptor(PresentOp op) : PresentOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PresentOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.present' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PresentOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PresentOp::getDataClauseAttrName(*odsOpName)) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PresentOp::getImplicitAttrName(*odsOpName)) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PresentOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PresentOp::getStructuredAttrName(*odsOpName)) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.present' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dataClause && !((tblgen_dataClause.isa<::mlir::acc::DataClauseAttr>())))
    return emitError(loc, "'acc.present' op ""attribute 'dataClause' failed to satisfy constraint: data clauses supported by OpenACC");

  if (tblgen_structured && !((tblgen_structured.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.present' op ""attribute 'structured' failed to satisfy constraint: bool attribute");

  if (tblgen_implicit && !((tblgen_implicit.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'acc.present' op ""attribute 'implicit' failed to satisfy constraint: bool attribute");

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'acc.present' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PresentOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PresentOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> PresentOp::getVarPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> PresentOp::getVarPtrPtr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::acc::PointerLikeType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*operands.begin());
}

::mlir::Operation::operand_range PresentOp::getBounds() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange PresentOp::getVarPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PresentOp::getVarPtrPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PresentOp::getBoundsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PresentOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PresentOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::acc::PointerLikeType> PresentOp::getAccPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::acc::PointerLikeType>>(*getODSResults(0).begin());
}

::mlir::acc::DataClauseAttr PresentOp::getDataClauseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDataClauseAttrName()).dyn_cast_or_null<::mlir::acc::DataClauseAttr>();
}

::mlir::acc::DataClause PresentOp::getDataClause() {
  auto attr = getDataClauseAttr();
  return attr.getValue();
}

::mlir::BoolAttr PresentOp::getStructuredAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStructuredAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool PresentOp::getStructured() {
  auto attr = getStructuredAttr();
  return attr.getValue();
}

::mlir::BoolAttr PresentOp::getImplicitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getImplicitAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool PresentOp::getImplicit() {
  auto attr = getImplicitAttr();
  return attr.getValue();
}

::mlir::StringAttr PresentOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > PresentOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void PresentOp::setDataClauseAttr(::mlir::acc::DataClauseAttr attr) {
  (*this)->setAttr(getDataClauseAttrName(), attr);
}

void PresentOp::setDataClause(::mlir::acc::DataClause attrValue) {
  (*this)->setAttr(getDataClauseAttrName(), ::mlir::acc::DataClauseAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void PresentOp::setStructuredAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getStructuredAttrName(), attr);
}

void PresentOp::setStructured(bool attrValue) {
  (*this)->setAttr(getStructuredAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void PresentOp::setImplicitAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getImplicitAttrName(), attr);
}

void PresentOp::setImplicit(bool attrValue) {
  (*this)->setAttr(getImplicitAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void PresentOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void PresentOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute PresentOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void PresentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void PresentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClauseAttr dataClause, ::mlir::BoolAttr structured, ::mlir::BoolAttr implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  if (dataClause) {
    odsState.addAttribute(getDataClauseAttrName(odsState.name), dataClause);
  }
  if (structured) {
    odsState.addAttribute(getStructuredAttrName(odsState.name), structured);
  }
  if (implicit) {
    odsState.addAttribute(getImplicitAttrName(odsState.name), implicit);
  }
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PresentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type accPtr, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  odsState.addTypes(accPtr);
}

void PresentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value varPtr, /*optional*/::mlir::Value varPtrPtr, ::mlir::ValueRange bounds, ::mlir::acc::DataClause dataClause, bool structured, bool implicit, /*optional*/::mlir::StringAttr name) {
  odsState.addOperands(varPtr);
  if (varPtrPtr)
    odsState.addOperands(varPtrPtr);
  odsState.addOperands(bounds);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (varPtrPtr ? 1 : 0), static_cast<int32_t>(bounds.size())}));
  odsState.addAttribute(getDataClauseAttrName(odsState.name), ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), dataClause));
  odsState.addAttribute(getStructuredAttrName(odsState.name), odsBuilder.getBoolAttr(structured));
  odsState.addAttribute(getImplicitAttrName(odsState.name), odsBuilder.getBoolAttr(implicit));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PresentOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PresentOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_present));
  }
  if (!attributes.get(attrNames[4])) {
    attributes.append(attrNames[4], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult PresentOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dataClause;
  ::mlir::Attribute tblgen_implicit;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDataClauseAttrName()) {
      tblgen_dataClause = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getImplicitAttrName()) {
      tblgen_implicit = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_structured;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStructuredAttrName()) {
      tblgen_structured = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_dataClause, "dataClause")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_structured, "structured")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_implicit, "implicit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PresentOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult PresentOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand varPtrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> varPtrOperands(varPtrRawOperands);  ::llvm::SMLoc varPtrOperandsLoc;
  (void)varPtrOperandsLoc;
  ::mlir::Type varPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> varPtrTypes(varPtrRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varPtrPtrOperands;
  ::llvm::SMLoc varPtrPtrOperandsLoc;
  (void)varPtrPtrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varPtrPtrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> boundsOperands;
  ::llvm::SMLoc boundsOperandsLoc;
  (void)boundsOperandsLoc;
  ::mlir::Type accPtrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accPtrTypes(accPtrRawTypes);
  if (parser.parseKeyword("varPtr"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  varPtrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(varPtrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    varPtrRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  bool varPtrPtrClause = false;
  bool boundsClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("varPtrPtr"))) {

  if (varPtrPtrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`varPtrPtr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  varPtrPtrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    varPtrPtrOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      varPtrPtrTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("bounds"))) {

  if (boundsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`bounds` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  boundsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  boundsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(boundsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::acc::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accPtrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(varPtrPtrOperands.size()), static_cast<int32_t>(boundsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::acc::DataBoundsType>();
  result.addTypes(accPtrTypes);
  if (parser.resolveOperands(varPtrOperands, varPtrTypes, varPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(varPtrPtrOperands, varPtrPtrTypes, varPtrPtrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(boundsOperands, odsBuildableType0, boundsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PresentOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "varPtr";
  _odsPrinter << "(";
  _odsPrinter << getVarPtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVarPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << " ";
  if (false || getVarPtrPtr()) {
  _odsPrinter << ' ' << "varPtrPtr";
  _odsPrinter << "(";
  if (::mlir::Value value = getVarPtrPtr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getVarPtrPtr() ? ::llvm::ArrayRef<::mlir::Type>(getVarPtrPtr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getBounds().size()) {
  _odsPrinter << ' ' << "bounds";
  _odsPrinter << "(";
  _odsPrinter << getBounds();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getAccPtr().getType();
    if (auto validType = type.dyn_cast<::mlir::acc::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataClauseAttr();
     if(attr && (attr == ::mlir::acc::DataClauseAttr::get(odsBuilder.getContext(), mlir::acc::DataClause::acc_present)))
       elidedAttrs.push_back("dataClause");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStructuredAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(true)))
       elidedAttrs.push_back("structured");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getImplicitAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("implicit");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::PresentOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::SerialOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SerialOpGenericAdaptorBase::SerialOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.serial", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SerialOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SerialOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SerialOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr SerialOpGenericAdaptorBase::getAsyncAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SerialOp::getAsyncAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool SerialOpGenericAdaptorBase::getAsyncAttr() {
  auto attr = getAsyncAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr SerialOpGenericAdaptorBase::getWaitAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SerialOp::getWaitAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool SerialOpGenericAdaptorBase::getWaitAttr() {
  auto attr = getWaitAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr SerialOpGenericAdaptorBase::getSelfAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SerialOp::getSelfAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool SerialOpGenericAdaptorBase::getSelfAttr() {
  auto attr = getSelfAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr SerialOpGenericAdaptorBase::getReductionOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SerialOp::getReductionOpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
  return attr;
}

::std::optional<::mlir::acc::ReductionOp> SerialOpGenericAdaptorBase::getReductionOp() {
  auto attr = getReductionOpAttr();
  return attr ? ::std::optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::std::nullopt);
}

::mlir::acc::ClauseDefaultValueAttr SerialOpGenericAdaptorBase::getDefaultAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SerialOp::getDefaultAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
  return attr;
}

::std::optional<::mlir::acc::ClauseDefaultValue> SerialOpGenericAdaptorBase::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &SerialOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange SerialOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
SerialOpAdaptor::SerialOpAdaptor(SerialOp op) : SerialOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SerialOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.serial' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SerialOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SerialOp::getAsyncAttrAttrName(*odsOpName)) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SerialOp::getDefaultAttrAttrName(*odsOpName)) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SerialOp::getReductionOpAttrName(*odsOpName)) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SerialOp::getSelfAttrAttrName(*odsOpName)) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SerialOp::getWaitAttrAttrName(*odsOpName)) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 19)
      return emitError(loc, "'acc.serial' op ""'operand_segment_sizes' attribute for specifying operand segments must have 19 "
                "elements, but got ") << numElements;
  }

  if (tblgen_asyncAttr && !((tblgen_asyncAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.serial' op ""attribute 'asyncAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_waitAttr && !((tblgen_waitAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.serial' op ""attribute 'waitAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_selfAttr && !((tblgen_selfAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.serial' op ""attribute 'selfAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_reductionOp && !((tblgen_reductionOp.isa<::mlir::acc::ReductionOpAttr>())))
    return emitError(loc, "'acc.serial' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");

  if (tblgen_defaultAttr && !((tblgen_defaultAttr.isa<::mlir::acc::ClauseDefaultValueAttr>())))
    return emitError(loc, "'acc.serial' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SerialOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SerialOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SerialOp::getAsync() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range SerialOp::getWaitOperands() {
  return getODSOperands(1);
}

::mlir::TypedValue<::mlir::IntegerType> SerialOp::getIfCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> SerialOp::getSelfCond() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range SerialOp::getReductionOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range SerialOp::getCopyOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range SerialOp::getCopyinOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range SerialOp::getCopyinReadonlyOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range SerialOp::getCopyoutOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range SerialOp::getCopyoutZeroOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range SerialOp::getCreateOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range SerialOp::getCreateZeroOperands() {
  return getODSOperands(11);
}

::mlir::Operation::operand_range SerialOp::getNoCreateOperands() {
  return getODSOperands(12);
}

::mlir::Operation::operand_range SerialOp::getPresentOperands() {
  return getODSOperands(13);
}

::mlir::Operation::operand_range SerialOp::getDevicePtrOperands() {
  return getODSOperands(14);
}

::mlir::Operation::operand_range SerialOp::getAttachOperands() {
  return getODSOperands(15);
}

::mlir::Operation::operand_range SerialOp::getGangPrivateOperands() {
  return getODSOperands(16);
}

::mlir::Operation::operand_range SerialOp::getGangFirstPrivateOperands() {
  return getODSOperands(17);
}

::mlir::Operation::operand_range SerialOp::getDataClauseOperands() {
  return getODSOperands(18);
}

::mlir::MutableOperandRange SerialOp::getAsyncMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getSelfCondMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getReductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCopyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCopyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCopyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCopyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCopyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getCreateZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getNoCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(12);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(12u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getPresentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(13);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(13u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getDevicePtrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(14);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(14u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getAttachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(15);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(15u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getGangPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(16);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(16u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getGangFirstPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(17);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(17u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SerialOp::getDataClauseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(18);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(18u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SerialOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SerialOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SerialOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr SerialOp::getAsyncAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SerialOp::getAsyncAttr() {
  auto attr = getAsyncAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr SerialOp::getWaitAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getWaitAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SerialOp::getWaitAttr() {
  auto attr = getWaitAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr SerialOp::getSelfAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSelfAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SerialOp::getSelfAttr() {
  auto attr = getSelfAttrAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr SerialOp::getReductionOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionOpAttrName()).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
}

::std::optional<::mlir::acc::ReductionOp> SerialOp::getReductionOp() {
  auto attr = getReductionOpAttr();
  return attr ? ::std::optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::std::nullopt);
}

::mlir::acc::ClauseDefaultValueAttr SerialOp::getDefaultAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDefaultAttrAttrName()).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
}

::std::optional<::mlir::acc::ClauseDefaultValue> SerialOp::getDefaultAttr() {
  auto attr = getDefaultAttrAttr();
  return attr ? ::std::optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::std::nullopt);
}

void SerialOp::setAsyncAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrAttrName(), attr);
}

void SerialOp::setAsyncAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrAttrName());
}

void SerialOp::setWaitAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWaitAttrAttrName(), attr);
}

void SerialOp::setWaitAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWaitAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWaitAttrAttrName());
}

void SerialOp::setSelfAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getSelfAttrAttrName(), attr);
}

void SerialOp::setSelfAttr(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSelfAttrAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getSelfAttrAttrName());
}

void SerialOp::setReductionOpAttr(::mlir::acc::ReductionOpAttr attr) {
  (*this)->setAttr(getReductionOpAttrName(), attr);
}

void SerialOp::setReductionOp(::std::optional<::mlir::acc::ReductionOp> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getReductionOpAttrName(), ::mlir::acc::ReductionOpAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getReductionOpAttrName());
}

void SerialOp::setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr) {
  (*this)->setAttr(getDefaultAttrAttrName(), attr);
}

void SerialOp::setDefaultAttr(::std::optional<::mlir::acc::ClauseDefaultValue> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDefaultAttrAttrName(), ::mlir::acc::ClauseDefaultValueAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getDefaultAttrAttrName());
}

::mlir::Attribute SerialOp::removeAsyncAttrAttr() {
  return (*this)->removeAttr(getAsyncAttrAttrName());
}

::mlir::Attribute SerialOp::removeWaitAttrAttr() {
  return (*this)->removeAttr(getWaitAttrAttrName());
}

::mlir::Attribute SerialOp::removeSelfAttrAttr() {
  return (*this)->removeAttr(getSelfAttrAttrName());
}

::mlir::Attribute SerialOp::removeReductionOpAttr() {
  return (*this)->removeAttr(getReductionOpAttrName());
}

::mlir::Attribute SerialOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr(getDefaultAttrAttrName());
}

void SerialOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), selfAttr);
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void SerialOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), selfAttr);
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SerialOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), ((asyncAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), ((waitAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), ((selfAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void SerialOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, ::mlir::ValueRange dataClauseOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addOperands(dataClauseOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size()), static_cast<int32_t>(dataClauseOperands.size())}));
  if (asyncAttr) {
    odsState.addAttribute(getAsyncAttrAttrName(odsState.name), ((asyncAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (waitAttr) {
    odsState.addAttribute(getWaitAttrAttrName(odsState.name), ((waitAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (selfAttr) {
    odsState.addAttribute(getSelfAttrAttrName(odsState.name), ((selfAttr) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (reductionOp) {
    odsState.addAttribute(getReductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
    odsState.addAttribute(getDefaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SerialOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SerialOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrAttrName()) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDefaultAttrAttrName()) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReductionOpAttrName()) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSelfAttrAttrName()) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getWaitAttrAttrName()) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 19)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 19 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_asyncAttr, "asyncAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_waitAttr, "waitAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_selfAttr, "selfAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps6(*this, tblgen_reductionOp, "reductionOp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps3(*this, tblgen_defaultAttr, "defaultAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup12 = getODSOperands(12);

    for (auto v : valueGroup12) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup13 = getODSOperands(13);

    for (auto v : valueGroup13) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup14 = getODSOperands(14);

    for (auto v : valueGroup14) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup15 = getODSOperands(15);

    for (auto v : valueGroup15) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup16 = getODSOperands(16);

    for (auto v : valueGroup16) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup17 = getODSOperands(17);

    for (auto v : valueGroup17) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup18 = getODSOperands(18);

    for (auto v : valueGroup18) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SerialOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SerialOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dataClauseOperandsOperands;
  ::llvm::SMLoc dataClauseOperandsOperandsLoc;
  (void)dataClauseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> dataClauseOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperands;
  ::llvm::SMLoc asyncOperandsLoc;
  (void)asyncOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyOperandsOperands;
  ::llvm::SMLoc copyOperandsOperandsLoc;
  (void)copyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinReadonlyOperandsOperands;
  ::llvm::SMLoc copyinReadonlyOperandsOperandsLoc;
  (void)copyinReadonlyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinReadonlyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutZeroOperandsOperands;
  ::llvm::SMLoc copyoutZeroOperandsOperandsLoc;
  (void)copyoutZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> devicePtrOperandsOperands;
  ::llvm::SMLoc devicePtrOperandsOperandsLoc;
  (void)devicePtrOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> devicePtrOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> gangFirstPrivateOperandsOperands;
  ::llvm::SMLoc gangFirstPrivateOperandsOperandsLoc;
  (void)gangFirstPrivateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> gangFirstPrivateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> noCreateOperandsOperands;
  ::llvm::SMLoc noCreateOperandsOperandsLoc;
  (void)noCreateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> noCreateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> gangPrivateOperandsOperands;
  ::llvm::SMLoc gangPrivateOperandsOperandsLoc;
  (void)gangPrivateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> gangPrivateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> presentOperandsOperands;
  ::llvm::SMLoc presentOperandsOperandsLoc;
  (void)presentOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> presentOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> selfCondOperands;
  ::llvm::SMLoc selfCondOperandsLoc;
  (void)selfCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reductionOperandsOperands;
  ::llvm::SMLoc reductionOperandsOperandsLoc;
  (void)reductionOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reductionOperandsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool dataOperandsClause = false;
  bool attachClause = false;
  bool asyncClause = false;
  bool copyClause = false;
  bool copyinClause = false;
  bool copyin_readonlyClause = false;
  bool copyoutClause = false;
  bool copyout_zeroClause = false;
  bool createClause = false;
  bool create_zeroClause = false;
  bool deviceptrClause = false;
  bool firstprivateClause = false;
  bool no_createClause = false;
  bool privateClause = false;
  bool presentClause = false;
  bool waitClause = false;
  bool selfClause = false;
  bool ifClause = false;
  bool reductionClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("dataOperands"))) {

  if (dataOperandsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`dataOperands` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dataOperandsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  dataClauseOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dataClauseOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(dataClauseOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("attach"))) {

  if (attachClause) {
    return parser.emitError(parser.getNameLoc())
          << "`attach` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  attachClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copy"))) {

  if (copyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copy` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin"))) {

  if (copyinClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyinClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyin_readonly"))) {

  if (copyin_readonlyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyin_readonly` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyin_readonlyClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinReadonlyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinReadonlyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinReadonlyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout"))) {

  if (copyoutClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyoutClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("copyout_zero"))) {

  if (copyout_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`copyout_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  copyout_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create"))) {

  if (createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("create_zero"))) {

  if (create_zeroClause) {
    return parser.emitError(parser.getNameLoc())
          << "`create_zero` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  create_zeroClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("deviceptr"))) {

  if (deviceptrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`deviceptr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deviceptrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  devicePtrOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(devicePtrOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(devicePtrOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("firstprivate"))) {

  if (firstprivateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`firstprivate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  firstprivateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  gangFirstPrivateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(gangFirstPrivateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(gangFirstPrivateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("no_create"))) {

  if (no_createClause) {
    return parser.emitError(parser.getNameLoc())
          << "`no_create` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  no_createClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  noCreateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(noCreateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(noCreateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("private"))) {

  if (privateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`private` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  privateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  gangPrivateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(gangPrivateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(gangPrivateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("present"))) {

  if (presentClause) {
    return parser.emitError(parser.getNameLoc())
          << "`present` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  presentClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  presentOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(presentOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(presentOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait"))) {

  if (waitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  waitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("self"))) {

  if (selfClause) {
    return parser.emitError(parser.getNameLoc())
          << "`self` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  selfClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    selfCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      selfCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  reductionOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(reductionOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(reductionOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(asyncOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(selfCondOperands.size()), static_cast<int32_t>(reductionOperandsOperands.size()), static_cast<int32_t>(copyOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(copyinReadonlyOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(copyoutZeroOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(noCreateOperandsOperands.size()), static_cast<int32_t>(presentOperandsOperands.size()), static_cast<int32_t>(devicePtrOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size()), static_cast<int32_t>(gangPrivateOperandsOperands.size()), static_cast<int32_t>(gangFirstPrivateOperandsOperands.size()), static_cast<int32_t>(dataClauseOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(asyncOperands, asyncTypes, asyncOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(selfCondOperands, odsBuildableType0, selfCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reductionOperandsOperands, reductionOperandsTypes, reductionOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyOperandsOperands, copyOperandsTypes, copyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinReadonlyOperandsOperands, copyinReadonlyOperandsTypes, copyinReadonlyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutZeroOperandsOperands, copyoutZeroOperandsTypes, copyoutZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(noCreateOperandsOperands, noCreateOperandsTypes, noCreateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(presentOperandsOperands, presentOperandsTypes, presentOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(devicePtrOperandsOperands, devicePtrOperandsTypes, devicePtrOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gangPrivateOperandsOperands, gangPrivateOperandsTypes, gangPrivateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gangFirstPrivateOperandsOperands, gangFirstPrivateOperandsTypes, gangFirstPrivateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataClauseOperandsOperands, dataClauseOperandsTypes, dataClauseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SerialOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getDataClauseOperands().size()) {
  _odsPrinter << ' ' << "dataOperands";
  _odsPrinter << "(";
  _odsPrinter << getDataClauseOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDataClauseOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAttachOperands().size()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << getAttachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getAttachOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getAsync()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsync())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsync() ? ::llvm::ArrayRef<::mlir::Type>(getAsync().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getCopyOperands().size()) {
  _odsPrinter << ' ' << "copy";
  _odsPrinter << "(";
  _odsPrinter << getCopyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinOperands().size()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << getCopyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyinReadonlyOperands().size()) {
  _odsPrinter << ' ' << "copyin_readonly";
  _odsPrinter << "(";
  _odsPrinter << getCopyinReadonlyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyinReadonlyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutOperands().size()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCopyoutZeroOperands().size()) {
  _odsPrinter << ' ' << "copyout_zero";
  _odsPrinter << "(";
  _odsPrinter << getCopyoutZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCopyoutZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateOperands().size()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << getCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getCreateZeroOperands().size()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << getCreateZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getCreateZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDevicePtrOperands().size()) {
  _odsPrinter << ' ' << "deviceptr";
  _odsPrinter << "(";
  _odsPrinter << getDevicePtrOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDevicePtrOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getGangFirstPrivateOperands().size()) {
  _odsPrinter << ' ' << "firstprivate";
  _odsPrinter << "(";
  _odsPrinter << getGangFirstPrivateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getGangFirstPrivateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getNoCreateOperands().size()) {
  _odsPrinter << ' ' << "no_create";
  _odsPrinter << "(";
  _odsPrinter << getNoCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getNoCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getGangPrivateOperands().size()) {
  _odsPrinter << ' ' << "private";
  _odsPrinter << "(";
  _odsPrinter << getGangPrivateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getGangPrivateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getPresentOperands().size()) {
  _odsPrinter << ' ' << "present";
  _odsPrinter << "(";
  _odsPrinter << getPresentOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getPresentOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getWaitOperands().size()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << getWaitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getWaitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getSelfCond()) {
  _odsPrinter << ' ' << "self";
  _odsPrinter << "(";
  if (::mlir::Value value = getSelfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getReductionOperands().size()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  _odsPrinter << getReductionOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getReductionOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("asyncAttr");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWaitAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("waitAttr");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSelfAttrAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("selfAttr");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::SerialOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ShutdownOpGenericAdaptorBase::ShutdownOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.shutdown", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ShutdownOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ShutdownOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ShutdownOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ShutdownOpAdaptor::ShutdownOpAdaptor(ShutdownOp op) : ShutdownOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ShutdownOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.shutdown' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ShutdownOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.shutdown' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> ShutdownOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ShutdownOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ShutdownOp::getDeviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value ShutdownOp::getDeviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> ShutdownOp::getIfCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::MutableOperandRange ShutdownOp::getDeviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ShutdownOp::getDeviceNumOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ShutdownOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ShutdownOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShutdownOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ShutdownOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
}

void ShutdownOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShutdownOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShutdownOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShutdownOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ShutdownOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceNumOperandOperands;
  ::llvm::SMLoc deviceNumOperandOperandsLoc;
  (void)deviceNumOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceNumOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  bool device_typeClause = false;
  bool device_numClause = false;
  bool ifClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("device_type"))) {

  if (device_typeClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device_type` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  device_typeClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("device_num"))) {

  if (device_numClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device_num` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  device_numClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceNumOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(deviceNumOperandOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceNumOperandOperands, deviceNumOperandTypes, deviceNumOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShutdownOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getDeviceTypeOperands().size()) {
  _odsPrinter << ' ' << "device_type";
  _odsPrinter << "(";
  _odsPrinter << getDeviceTypeOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDeviceTypeOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDeviceNumOperand()) {
  _odsPrinter << ' ' << "device_num";
  _odsPrinter << "(";
  if (::mlir::Value value = getDeviceNumOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getDeviceNumOperand() ? ::llvm::ArrayRef<::mlir::Type>(getDeviceNumOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ShutdownOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TerminatorOpGenericAdaptorBase::TerminatorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.terminator", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TerminatorOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TerminatorOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TerminatorOpAdaptor::TerminatorOpAdaptor(TerminatorOp op) : TerminatorOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TerminatorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TerminatorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TerminatorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TerminatorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TerminatorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TerminatorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TerminatorOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TerminatorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TerminatorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TerminatorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::TerminatorOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UpdateOpGenericAdaptorBase::UpdateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.update", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UpdateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, UpdateOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr UpdateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr UpdateOpGenericAdaptorBase::getAsyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, UpdateOp::getAsyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool UpdateOpGenericAdaptorBase::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOpGenericAdaptorBase::getWaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, UpdateOp::getWaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool UpdateOpGenericAdaptorBase::getWait() {
  auto attr = getWaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOpGenericAdaptorBase::getIfPresentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, UpdateOp::getIfPresentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool UpdateOpGenericAdaptorBase::getIfPresent() {
  auto attr = getIfPresentAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
UpdateOpAdaptor::UpdateOpAdaptor(UpdateOp op) : UpdateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UpdateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_ifPresent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.update' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == UpdateOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == UpdateOp::getAsyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == UpdateOp::getIfPresentAttrName(*odsOpName)) {
      tblgen_ifPresent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == UpdateOp::getWaitAttrName(*odsOpName)) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'acc.update' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.update' op ""attribute 'async' failed to satisfy constraint: unit attribute");

  if (tblgen_wait && !((tblgen_wait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.update' op ""attribute 'wait' failed to satisfy constraint: unit attribute");

  if (tblgen_ifPresent && !((tblgen_ifPresent.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.update' op ""attribute 'ifPresent' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UpdateOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range UpdateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> UpdateOp::getIfCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Value UpdateOp::getAsyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value UpdateOp::getWaitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range UpdateOp::getWaitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range UpdateOp::getDeviceTypeOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range UpdateOp::getHostOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range UpdateOp::getDeviceOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange UpdateOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::getAsyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::getWaitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::getDeviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::getHostOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::getDeviceOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> UpdateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UpdateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr UpdateOp::getAsyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOp::getWaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getWaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::getWait() {
  auto attr = getWaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOp::getIfPresentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIfPresentAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::getIfPresent() {
  auto attr = getIfPresentAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void UpdateOp::setAsyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrName(), attr);
}

void UpdateOp::setAsync(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrName());
}

void UpdateOp::setWaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWaitAttrName(), attr);
}

void UpdateOp::setWait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWaitAttrName());
}

void UpdateOp::setIfPresentAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIfPresentAttrName(), attr);
}

void UpdateOp::setIfPresent(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIfPresentAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getIfPresentAttrName());
}

::mlir::Attribute UpdateOp::removeAsyncAttr() {
  return (*this)->removeAttr(getAsyncAttrName());
}

::mlir::Attribute UpdateOp::removeWaitAttr() {
  return (*this)->removeAttr(getWaitAttrName());
}

::mlir::Attribute UpdateOp::removeIfPresentAttr() {
  return (*this)->removeAttr(getIfPresentAttrName());
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), wait);
  }
  if (ifPresent) {
    odsState.addAttribute(getIfPresentAttrName(odsState.name), ifPresent);
  }
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), wait);
  }
  if (ifPresent) {
    odsState.addAttribute(getIfPresentAttrName(odsState.name), ifPresent);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), ((wait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (ifPresent) {
    odsState.addAttribute(getIfPresentAttrName(odsState.name), ((ifPresent) ? odsBuilder.getUnitAttr() : nullptr));
  }
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (wait) {
    odsState.addAttribute(getWaitAttrName(odsState.name), ((wait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (ifPresent) {
    odsState.addAttribute(getIfPresentAttrName(odsState.name), ((ifPresent) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UpdateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_ifPresent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIfPresentAttrName()) {
      tblgen_ifPresent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getWaitAttrName()) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_async, "async")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_wait, "wait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_ifPresent, "ifPresent")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UpdateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult UpdateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> hostOperandsOperands;
  ::llvm::SMLoc hostOperandsOperandsLoc;
  (void)hostOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> hostOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperandsOperands;
  ::llvm::SMLoc deviceOperandsOperandsLoc;
  (void)deviceOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceOperandsTypes;
  bool ifClause = false;
  bool asyncClause = false;
  bool wait_devnumClause = false;
  bool device_typeClause = false;
  bool waitClause = false;
  bool hostClause = false;
  bool deviceClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {

  if (wait_devnumClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait_devnum` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  wait_devnumClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("device_type"))) {

  if (device_typeClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device_type` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  device_typeClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait"))) {

  if (waitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  waitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("host"))) {

  if (hostClause) {
    return parser.emitError(parser.getNameLoc())
          << "`host` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hostClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  hostOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(hostOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(hostOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("device"))) {

  if (deviceClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deviceClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(hostOperandsOperands.size()), static_cast<int32_t>(deviceOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(hostOperandsOperands, hostOperandsTypes, hostOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperandsOperands, deviceOperandsTypes, deviceOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UpdateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getAsyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(getAsyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = getWaitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getWaitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(getWaitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getDeviceTypeOperands().size()) {
  _odsPrinter << ' ' << "device_type";
  _odsPrinter << "(";
  _odsPrinter << getDeviceTypeOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDeviceTypeOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getWaitOperands().size()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << getWaitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getWaitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getHostOperands().size()) {
  _odsPrinter << ' ' << "host";
  _odsPrinter << "(";
  _odsPrinter << getHostOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getHostOperands().getTypes();
  _odsPrinter << ")";
  }
  if (false || getDeviceOperands().size()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  _odsPrinter << getDeviceOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDeviceOperands().getTypes();
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("async");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("wait");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIfPresentAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("ifPresent");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::UpdateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WaitOpGenericAdaptorBase::WaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.wait", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> WaitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, WaitOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr WaitOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr WaitOpGenericAdaptorBase::getAsyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WaitOp::getAsyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool WaitOpGenericAdaptorBase::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
WaitOpAdaptor::WaitOpAdaptor(WaitOp op) : WaitOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult WaitOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.wait' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == WaitOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WaitOp::getAsyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'acc.wait' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.wait' op ""attribute 'async' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WaitOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range WaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WaitOp::getWaitOperands() {
  return getODSOperands(0);
}

::mlir::Value WaitOp::getAsyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value WaitOp::getWaitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> WaitOp::getIfCond() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::MutableOperandRange WaitOp::getWaitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WaitOp::getAsyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WaitOp::getWaitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WaitOp::getIfCondMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> WaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr WaitOp::getAsyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WaitOp::getAsync() {
  auto attr = getAsyncAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void WaitOp::setAsyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getAsyncAttrName(), attr);
}

void WaitOp::setAsync(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsyncAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getAsyncAttrName());
}

::mlir::Attribute WaitOp::removeAsyncAttr() {
  return (*this)->removeAttr(getAsyncAttrName());
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), async);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
    odsState.addAttribute(getAsyncAttrName(odsState.name), ((async) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WaitOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps4(*this, tblgen_async, "async")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WaitOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  bool asyncClause = false;
  bool wait_devnumClause = false;
  bool ifClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("async"))) {

  if (asyncClause) {
    return parser.emitError(parser.getNameLoc())
          << "`async` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  asyncClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {

  if (wait_devnumClause) {
    return parser.emitError(parser.getNameLoc())
          << "`wait_devnum` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  wait_devnumClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getWaitOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getWaitOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getWaitOperands().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ' ' << " ";
  if (false || getAsyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = getAsyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getAsyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(getAsyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getWaitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = getWaitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getWaitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(getWaitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getIfCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAsyncAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("async");
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::WaitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
YieldOpGenericAdaptorBase::YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.yield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> YieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr YieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : YieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::getOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 /* nothing to do */ 
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getOperands().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getOperands().getTypes();
  }
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::YieldOp)


#endif  // GET_OP_CLASSES

