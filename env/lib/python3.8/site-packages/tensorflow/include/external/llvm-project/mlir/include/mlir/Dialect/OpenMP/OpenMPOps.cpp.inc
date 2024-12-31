/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::omp::AtomicCaptureOp,
::mlir::omp::AtomicReadOp,
::mlir::omp::AtomicUpdateOp,
::mlir::omp::AtomicWriteOp,
::mlir::omp::BarrierOp,
::mlir::omp::CancelOp,
::mlir::omp::CancellationPointOp,
::mlir::omp::CriticalDeclareOp,
::mlir::omp::CriticalOp,
::mlir::omp::FlushOp,
::mlir::omp::MasterOp,
::mlir::omp::OrderedOp,
::mlir::omp::OrderedRegionOp,
::mlir::omp::ParallelOp,
::mlir::omp::ReductionDeclareOp,
::mlir::omp::ReductionOp,
::mlir::omp::SectionOp,
::mlir::omp::SectionsOp,
::mlir::omp::SimdLoopOp,
::mlir::omp::SingleOp,
::mlir::omp::TargetOp,
::mlir::omp::DataOp,
::mlir::omp::EnterDataOp,
::mlir::omp::ExitDataOp,
::mlir::omp::TaskGroupOp,
::mlir::omp::TaskLoopOp,
::mlir::omp::TaskOp,
::mlir::omp::TaskwaitOp,
::mlir::omp::TaskyieldOp,
::mlir::omp::TerminatorOp,
::mlir::omp::ThreadprivateOp,
::mlir::omp::WsLoopOp,
::mlir::omp::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::omp::PointerLikeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be OpenMP-compatible variable type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseMemoryOrderKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: MemoryOrderKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseCancellationConstructTypeAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: CancellationConstructType Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseDependAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: depend clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getInt() >= 0)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseProcBindKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ProcBindKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseOrderKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: OrderKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::omp::ClauseTaskDependAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: clause_task_depend array attr";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseScheduleKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ScheduleKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ScheduleModifierAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: OpenMP Schedule Modifier";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_OpenMPOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_OpenMPOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicCaptureOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicCaptureOpGenericAdaptorBase::AtomicCaptureOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.capture", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicCaptureOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicCaptureOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicCaptureOpGenericAdaptorBase::getHintValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicCaptureOp::getHintValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

uint64_t AtomicCaptureOpGenericAdaptorBase::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicCaptureOpGenericAdaptorBase::getMemoryOrderValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicCaptureOp::getMemoryOrderValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicCaptureOpGenericAdaptorBase::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &AtomicCaptureOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange AtomicCaptureOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
AtomicCaptureOpAdaptor::AtomicCaptureOpAdaptor(AtomicCaptureOp op) : AtomicCaptureOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicCaptureOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicCaptureOp::getHintValAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCaptureOp::getMemoryOrderValAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.capture' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.capture' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicCaptureOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicCaptureOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AtomicCaptureOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicCaptureOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &AtomicCaptureOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr AtomicCaptureOp::getHintValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getHintValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicCaptureOp::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicCaptureOp::getMemoryOrderValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMemoryOrderValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicCaptureOp::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

void AtomicCaptureOp::setHintValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getHintValAttrName(), attr);
}

void AtomicCaptureOp::setHintVal(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHintValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getHintValAttrName());
}

void AtomicCaptureOp::setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(getMemoryOrderValAttrName(), attr);
}

void AtomicCaptureOp::setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getMemoryOrderValAttrName(), ::mlir::omp::ClauseMemoryOrderKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getMemoryOrderValAttrName());
}

::mlir::Attribute AtomicCaptureOp::removeHintValAttr() {
  return (*this)->removeAttr(getHintValAttrName());
}

::mlir::Attribute AtomicCaptureOp::removeMemoryOrderValAttr() {
  return (*this)->removeAttr(getMemoryOrderValAttrName());
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCaptureOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicCaptureOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHintValAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMemoryOrderValAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicCaptureOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicCaptureOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::IntegerAttr hint_valAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool memory_orderClause = false;
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    if (hint_valAttr)
      result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();

  ensureTerminator(*regionRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void AtomicCaptureOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getMemoryOrderValAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getMemoryOrderValAttr());
  _odsPrinter << ")";
  }
  if (false || getHintValAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, getHintValAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getRegion().empty() ? nullptr : getRegion().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("memory_order_val");
  elidedAttrs.push_back("hint_val");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHintValAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("hint_val");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicCaptureOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicReadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicReadOpGenericAdaptorBase::AtomicReadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.read", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicReadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicReadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr AtomicReadOpGenericAdaptorBase::getElementTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicReadOp::getElementTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type AtomicReadOpGenericAdaptorBase::getElementType() {
  auto attr = getElementTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::IntegerAttr AtomicReadOpGenericAdaptorBase::getHintValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AtomicReadOp::getHintValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

uint64_t AtomicReadOpGenericAdaptorBase::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicReadOpGenericAdaptorBase::getMemoryOrderValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AtomicReadOp::getMemoryOrderValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicReadOpGenericAdaptorBase::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
AtomicReadOpAdaptor::AtomicReadOpAdaptor(AtomicReadOp op) : AtomicReadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicReadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_element_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.atomic.read' op ""requires attribute 'element_type'");
    if (namedAttrIt->getName() == AtomicReadOp::getElementTypeAttrName(*odsOpName)) {
      tblgen_element_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicReadOp::getHintValAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicReadOp::getMemoryOrderValAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_element_type && !(((tblgen_element_type.isa<::mlir::TypeAttr>())) && ((tblgen_element_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'omp.atomic.read' op ""attribute 'element_type' failed to satisfy constraint: any type attribute");

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.read' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.read' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicReadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicReadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> AtomicReadOp::getX() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> AtomicReadOp::getV() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AtomicReadOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicReadOp::getVMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicReadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicReadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypeAttr AtomicReadOp::getElementTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElementTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type AtomicReadOp::getElementType() {
  auto attr = getElementTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::IntegerAttr AtomicReadOp::getHintValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getHintValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicReadOp::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicReadOp::getMemoryOrderValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getMemoryOrderValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicReadOp::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

void AtomicReadOp::setElementTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getElementTypeAttrName(), attr);
}

void AtomicReadOp::setElementType(::mlir::Type attrValue) {
  (*this)->setAttr(getElementTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void AtomicReadOp::setHintValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getHintValAttrName(), attr);
}

void AtomicReadOp::setHintVal(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHintValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getHintValAttrName());
}

void AtomicReadOp::setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(getMemoryOrderValAttrName(), attr);
}

void AtomicReadOp::setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getMemoryOrderValAttrName(), ::mlir::omp::ClauseMemoryOrderKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getMemoryOrderValAttrName());
}

::mlir::Attribute AtomicReadOp::removeHintValAttr() {
  return (*this)->removeAttr(getHintValAttrName());
}

::mlir::Attribute AtomicReadOp::removeMemoryOrderValAttr() {
  return (*this)->removeAttr(getMemoryOrderValAttrName());
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, ::mlir::TypeAttr element_type, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(getElementTypeAttrName(odsState.name), element_type);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::TypeAttr element_type, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(getElementTypeAttrName(odsState.name), element_type);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, ::mlir::Type element_type, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(getElementTypeAttrName(odsState.name), ::mlir::TypeAttr::get(element_type));
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::Type element_type, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(getElementTypeAttrName(odsState.name), ::mlir::TypeAttr::get(element_type));
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicReadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicReadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_element_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'element_type'");
    if (namedAttrIt->getName() == getElementTypeAttrName()) {
      tblgen_element_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHintValAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMemoryOrderValAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps2(*this, tblgen_element_type, "element_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()) && ((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {x, v} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult AtomicReadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicReadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand vRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> vOperands(vRawOperands);  ::llvm::SMLoc vOperandsLoc;
  (void)vOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::IntegerAttr hint_valAttr;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  ::mlir::TypeAttr element_typeAttr;

  vOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  bool memory_orderClause = false;
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    if (hint_valAttr)
      result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(element_typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "element_type",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(vOperands, xTypes[0], vOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicReadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getV();
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter << ' ' << " ";
  if (false || getMemoryOrderValAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getMemoryOrderValAttr());
  _odsPrinter << ")";
  }
  if (false || getHintValAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, getHintValAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getElementTypeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("memory_order_val");
  elidedAttrs.push_back("hint_val");
  elidedAttrs.push_back("element_type");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHintValAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("hint_val");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicReadOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicUpdateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicUpdateOpGenericAdaptorBase::AtomicUpdateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.update", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicUpdateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicUpdateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicUpdateOpGenericAdaptorBase::getHintValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicUpdateOp::getHintValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

uint64_t AtomicUpdateOpGenericAdaptorBase::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicUpdateOpGenericAdaptorBase::getMemoryOrderValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicUpdateOp::getMemoryOrderValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicUpdateOpGenericAdaptorBase::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &AtomicUpdateOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange AtomicUpdateOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
AtomicUpdateOpAdaptor::AtomicUpdateOpAdaptor(AtomicUpdateOp op) : AtomicUpdateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicUpdateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicUpdateOp::getHintValAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicUpdateOp::getMemoryOrderValAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.update' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.update' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicUpdateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicUpdateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> AtomicUpdateOp::getX() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange AtomicUpdateOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicUpdateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicUpdateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &AtomicUpdateOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr AtomicUpdateOp::getHintValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getHintValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicUpdateOp::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicUpdateOp::getMemoryOrderValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMemoryOrderValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicUpdateOp::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

void AtomicUpdateOp::setHintValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getHintValAttrName(), attr);
}

void AtomicUpdateOp::setHintVal(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHintValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getHintValAttrName());
}

void AtomicUpdateOp::setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(getMemoryOrderValAttrName(), attr);
}

void AtomicUpdateOp::setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getMemoryOrderValAttrName(), ::mlir::omp::ClauseMemoryOrderKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getMemoryOrderValAttrName());
}

::mlir::Attribute AtomicUpdateOp::removeHintValAttr() {
  return (*this)->removeAttr(getHintValAttrName());
}

::mlir::Attribute AtomicUpdateOp::removeMemoryOrderValAttr() {
  return (*this)->removeAttr(getMemoryOrderValAttrName());
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicUpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicUpdateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHintValAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMemoryOrderValAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicUpdateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AtomicUpdateOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::ParseResult AtomicUpdateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::IntegerAttr hint_valAttr;
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool memory_orderClause = false;
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    if (hint_valAttr)
      result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();

  ensureTerminator(*regionRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicUpdateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getMemoryOrderValAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getMemoryOrderValAttr());
  _odsPrinter << ")";
  }
  if (false || getHintValAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, getHintValAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getRegion().empty() ? nullptr : getRegion().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("memory_order_val");
  elidedAttrs.push_back("hint_val");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHintValAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("hint_val");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void AtomicUpdateOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicUpdateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicWriteOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicWriteOpGenericAdaptorBase::AtomicWriteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.write", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicWriteOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicWriteOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicWriteOpGenericAdaptorBase::getHintValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicWriteOp::getHintValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

uint64_t AtomicWriteOpGenericAdaptorBase::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicWriteOpGenericAdaptorBase::getMemoryOrderValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicWriteOp::getMemoryOrderValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicWriteOpGenericAdaptorBase::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
AtomicWriteOpAdaptor::AtomicWriteOpAdaptor(AtomicWriteOp op) : AtomicWriteOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicWriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicWriteOp::getHintValAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicWriteOp::getMemoryOrderValAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.write' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.write' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicWriteOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicWriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> AtomicWriteOp::getAddress() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::Value AtomicWriteOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AtomicWriteOp::getAddressMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicWriteOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicWriteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicWriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AtomicWriteOp::getHintValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getHintValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicWriteOp::getHintVal() {
  auto attr = getHintValAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicWriteOp::getMemoryOrderValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMemoryOrderValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::std::optional<::mlir::omp::ClauseMemoryOrderKind> AtomicWriteOp::getMemoryOrderVal() {
  auto attr = getMemoryOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::std::nullopt);
}

void AtomicWriteOp::setHintValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getHintValAttrName(), attr);
}

void AtomicWriteOp::setHintVal(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHintValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getHintValAttrName());
}

void AtomicWriteOp::setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(getMemoryOrderValAttrName(), attr);
}

void AtomicWriteOp::setMemoryOrderVal(::std::optional<::mlir::omp::ClauseMemoryOrderKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getMemoryOrderValAttrName(), ::mlir::omp::ClauseMemoryOrderKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getMemoryOrderValAttrName());
}

::mlir::Attribute AtomicWriteOp::removeHintValAttr() {
  return (*this)->removeAttr(getHintValAttrName());
}

::mlir::Attribute AtomicWriteOp::removeMemoryOrderValAttr() {
  return (*this)->removeAttr(getMemoryOrderValAttrName());
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
    odsState.addAttribute(getMemoryOrderValAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicWriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicWriteOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHintValAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMemoryOrderValAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicWriteOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicWriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand addressRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addressOperands(addressRawOperands);  ::llvm::SMLoc addressOperandsLoc;
  (void)addressOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::IntegerAttr hint_valAttr;
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::Type addressRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> addressTypes(addressRawTypes);
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  addressOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addressRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  bool hintClause = false;
  bool memory_orderClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    if (hint_valAttr)
      result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addressRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(addressOperands, addressTypes, addressOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicWriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAddress();
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << " ";
  if (false || getHintValAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, getHintValAttr());
  _odsPrinter << ")";
  }
  if (false || getMemoryOrderValAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getMemoryOrderValAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAddress().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("hint_val");
  elidedAttrs.push_back("memory_order_val");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHintValAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("hint_val");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicWriteOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BarrierOpGenericAdaptorBase::BarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.barrier", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BarrierOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BarrierOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BarrierOpAdaptor::BarrierOpAdaptor(BarrierOp op) : BarrierOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BarrierOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BarrierOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BarrierOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BarrierOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void BarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void BarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BarrierOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult BarrierOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CancelOpGenericAdaptorBase::CancelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.cancel", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CancelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr CancelOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancelOpGenericAdaptorBase::getCancellationConstructTypeValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CancelOp::getCancellationConstructTypeValAttrName(*odsOpName)).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
  return attr;
}

::mlir::omp::ClauseCancellationConstructType CancelOpGenericAdaptorBase::getCancellationConstructTypeVal() {
  auto attr = getCancellationConstructTypeValAttr();
  return attr.getValue();
}

} // namespace detail
CancelOpAdaptor::CancelOpAdaptor(CancelOp op) : CancelOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CancelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.cancel' op ""requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == CancelOp::getCancellationConstructTypeValAttrName(*odsOpName)) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_cancellation_construct_type_val && !((tblgen_cancellation_construct_type_val.isa<::mlir::omp::ClauseCancellationConstructTypeAttr>())))
    return emitError(loc, "'omp.cancel' op ""attribute 'cancellation_construct_type_val' failed to satisfy constraint: CancellationConstructType Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CancelOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CancelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> CancelOp::getIfExpr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::MutableOperandRange CancelOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CancelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CancelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancelOp::getCancellationConstructTypeValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCancellationConstructTypeValAttrName()).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
}

::mlir::omp::ClauseCancellationConstructType CancelOp::getCancellationConstructTypeVal() {
  auto attr = getCancellationConstructTypeValAttr();
  return attr.getValue();
}

void CancelOp::setCancellationConstructTypeValAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr) {
  (*this)->setAttr(getCancellationConstructTypeValAttrName(), attr);
}

void CancelOp::setCancellationConstructTypeVal(::mlir::omp::ClauseCancellationConstructType attrValue) {
  (*this)->setAttr(getCancellationConstructTypeValAttrName(), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), cancellation_construct_type_val);
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), cancellation_construct_type_val);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CancelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == getCancellationConstructTypeValAttrName()) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps3(*this, tblgen_cancellation_construct_type_val, "cancellation_construct_type_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CancelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CancelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_valAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  if (parser.parseKeyword("cancellation_construct_type"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, cancellation_construct_type_valAttr))
      return ::mlir::failure();
    result.addAttribute("cancellation_construct_type_val", cancellation_construct_type_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CancelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "cancellation_construct_type";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getCancellationConstructTypeValAttr());
  _odsPrinter << ")";
  if (getIfExpr()) {
    _odsPrinter << ' ' << "if";
    _odsPrinter << "(";
    if (::mlir::Value value = getIfExpr())
      _odsPrinter << value;
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("cancellation_construct_type_val");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CancelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancellationPointOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CancellationPointOpGenericAdaptorBase::CancellationPointOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.cancellationpoint", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CancellationPointOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CancellationPointOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancellationPointOpGenericAdaptorBase::getCancellationConstructTypeValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CancellationPointOp::getCancellationConstructTypeValAttrName(*odsOpName)).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
  return attr;
}

::mlir::omp::ClauseCancellationConstructType CancellationPointOpGenericAdaptorBase::getCancellationConstructTypeVal() {
  auto attr = getCancellationConstructTypeValAttr();
  return attr.getValue();
}

} // namespace detail
CancellationPointOpAdaptor::CancellationPointOpAdaptor(CancellationPointOp op) : CancellationPointOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CancellationPointOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.cancellationpoint' op ""requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == CancellationPointOp::getCancellationConstructTypeValAttrName(*odsOpName)) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_cancellation_construct_type_val && !((tblgen_cancellation_construct_type_val.isa<::mlir::omp::ClauseCancellationConstructTypeAttr>())))
    return emitError(loc, "'omp.cancellationpoint' op ""attribute 'cancellation_construct_type_val' failed to satisfy constraint: CancellationConstructType Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CancellationPointOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CancellationPointOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CancellationPointOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CancellationPointOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancellationPointOp::getCancellationConstructTypeValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCancellationConstructTypeValAttrName()).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
}

::mlir::omp::ClauseCancellationConstructType CancellationPointOp::getCancellationConstructTypeVal() {
  auto attr = getCancellationConstructTypeValAttr();
  return attr.getValue();
}

void CancellationPointOp::setCancellationConstructTypeValAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr) {
  (*this)->setAttr(getCancellationConstructTypeValAttrName(), attr);
}

void CancellationPointOp::setCancellationConstructTypeVal(::mlir::omp::ClauseCancellationConstructType attrValue) {
  (*this)->setAttr(getCancellationConstructTypeValAttrName(), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val) {
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), cancellation_construct_type_val);
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val) {
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), cancellation_construct_type_val);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val) {
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val) {
  odsState.addAttribute(getCancellationConstructTypeValAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancellationPointOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CancellationPointOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == getCancellationConstructTypeValAttrName()) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps3(*this, tblgen_cancellation_construct_type_val, "cancellation_construct_type_val")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult CancellationPointOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CancellationPointOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_valAttr;
  if (parser.parseKeyword("cancellation_construct_type"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, cancellation_construct_type_valAttr))
      return ::mlir::failure();
    result.addAttribute("cancellation_construct_type_val", cancellation_construct_type_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CancellationPointOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "cancellation_construct_type";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getCancellationConstructTypeValAttr());
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("cancellation_construct_type_val");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CancellationPointOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CriticalDeclareOpGenericAdaptorBase::CriticalDeclareOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.critical.declare", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CriticalDeclareOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CriticalDeclareOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CriticalDeclareOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CriticalDeclareOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CriticalDeclareOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr CriticalDeclareOpGenericAdaptorBase::getHintValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CriticalDeclareOp::getHintValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t CriticalDeclareOpGenericAdaptorBase::getHintVal() {
  auto attr = getHintValAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(CriticalDeclareOp op) : CriticalDeclareOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CriticalDeclareOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_hint_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.critical.declare' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == CriticalDeclareOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CriticalDeclareOp::getHintValAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'omp.critical.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.critical.declare' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CriticalDeclareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CriticalDeclareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CriticalDeclareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CriticalDeclareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr CriticalDeclareOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CriticalDeclareOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr CriticalDeclareOp::getHintValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getHintValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t CriticalDeclareOp::getHintVal() {
  auto attr = getHintValAttr();
  return attr.getValue().getZExtValue();
}

void CriticalDeclareOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void CriticalDeclareOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CriticalDeclareOp::setHintValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getHintValAttrName(), attr);
}

void CriticalDeclareOp::setHintVal(uint64_t attrValue) {
  (*this)->setAttr(getHintValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (hint_val) {
    odsState.addAttribute(getHintValAttrName(odsState.name), hint_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint64_t hint_val) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t hint_val) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getHintValAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0));
  }
}

::mlir::LogicalResult CriticalDeclareOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_hint_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getHintValAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps4(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult CriticalDeclareOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CriticalDeclareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::IntegerAttr hint_valAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    if (hint_valAttr)
      result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CriticalDeclareOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter << ' ' << " ";
  if (false || getHintValAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, getHintValAttr());
  _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  elidedAttrs.push_back("hint_val");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHintValAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("hint_val");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CriticalOpGenericAdaptorBase::CriticalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.critical", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CriticalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CriticalOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr CriticalOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CriticalOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > CriticalOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &CriticalOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange CriticalOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
CriticalOpAdaptor::CriticalOpAdaptor(CriticalOp op) : CriticalOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CriticalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CriticalOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'omp.critical' op ""attribute 'name' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CriticalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CriticalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CriticalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CriticalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &CriticalOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::FlatSymbolRefAttr CriticalOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::std::optional< ::llvm::StringRef > CriticalOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void CriticalOp::setNameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CriticalOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute CriticalOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void CriticalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name) {
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
}

void CriticalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name) {
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CriticalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps5(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult CriticalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CriticalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr nameAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void CriticalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("name")) {
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getNameAttr());
    _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FlushOpGenericAdaptorBase::FlushOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.flush", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FlushOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr FlushOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FlushOpAdaptor::FlushOpAdaptor(FlushOp op) : FlushOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FlushOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FlushOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range FlushOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FlushOp::getVarList() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange FlushOp::getVarListMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FlushOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FlushOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FlushOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList) {
  odsState.addOperands(varList);
}

void FlushOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FlushOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FlushOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FlushOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varListOperands;
  ::llvm::SMLoc varListOperandsLoc;
  (void)varListOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varListTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  varListOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(varListOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(varListTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(varListOperands, varListTypes, varListOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FlushOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getVarList().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getVarList();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getVarList().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MasterOpGenericAdaptorBase::MasterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.master", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MasterOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MasterOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &MasterOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange MasterOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MasterOpAdaptor::MasterOpAdaptor(MasterOp op) : MasterOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MasterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MasterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MasterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> MasterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MasterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &MasterOp::getRegion() {
  return (*this)->getRegion(0);
}

void MasterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
  (void)odsState.addRegion();
}

void MasterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MasterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MasterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MasterOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MasterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void MasterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
OrderedOpGenericAdaptorBase::OrderedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.ordered", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> OrderedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr OrderedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseDependAttr OrderedOpGenericAdaptorBase::getDependTypeValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OrderedOp::getDependTypeValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseDependAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseDepend> OrderedOpGenericAdaptorBase::getDependTypeVal() {
  auto attr = getDependTypeValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseDepend>(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr OrderedOpGenericAdaptorBase::getNumLoopsValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OrderedOp::getNumLoopsValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> OrderedOpGenericAdaptorBase::getNumLoopsVal() {
  auto attr = getNumLoopsValAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
OrderedOpAdaptor::OrderedOpAdaptor(OrderedOp op) : OrderedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult OrderedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_depend_type_val;
  ::mlir::Attribute tblgen_num_loops_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == OrderedOp::getDependTypeValAttrName(*odsOpName)) {
      tblgen_depend_type_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OrderedOp::getNumLoopsValAttrName(*odsOpName)) {
      tblgen_num_loops_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_depend_type_val && !((tblgen_depend_type_val.isa<::mlir::omp::ClauseDependAttr>())))
    return emitError(loc, "'omp.ordered' op ""attribute 'depend_type_val' failed to satisfy constraint: depend clause");

  if (tblgen_num_loops_val && !((((tblgen_num_loops_val.isa<::mlir::IntegerAttr>())) && ((tblgen_num_loops_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_num_loops_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'omp.ordered' op ""attribute 'num_loops_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OrderedOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OrderedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OrderedOp::getDependVecVars() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange OrderedOp::getDependVecVarsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OrderedOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrderedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::omp::ClauseDependAttr OrderedOp::getDependTypeValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDependTypeValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseDependAttr>();
}

::std::optional<::mlir::omp::ClauseDepend> OrderedOp::getDependTypeVal() {
  auto attr = getDependTypeValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseDepend>(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr OrderedOp::getNumLoopsValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNumLoopsValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> OrderedOp::getNumLoopsVal() {
  auto attr = getNumLoopsValAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void OrderedOp::setDependTypeValAttr(::mlir::omp::ClauseDependAttr attr) {
  (*this)->setAttr(getDependTypeValAttrName(), attr);
}

void OrderedOp::setDependTypeVal(::std::optional<::mlir::omp::ClauseDepend> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDependTypeValAttrName(), ::mlir::omp::ClauseDependAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getDependTypeValAttrName());
}

void OrderedOp::setNumLoopsValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumLoopsValAttrName(), attr);
}

void OrderedOp::setNumLoopsVal(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNumLoopsValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getNumLoopsValAttrName());
}

::mlir::Attribute OrderedOp::removeDependTypeValAttr() {
  return (*this)->removeAttr(getDependTypeValAttrName());
}

::mlir::Attribute OrderedOp::removeNumLoopsValAttr() {
  return (*this)->removeAttr(getNumLoopsValAttrName());
}

void OrderedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars) {
  odsState.addOperands(depend_vec_vars);
  if (depend_type_val) {
    odsState.addAttribute(getDependTypeValAttrName(odsState.name), depend_type_val);
  }
  if (num_loops_val) {
    odsState.addAttribute(getNumLoopsValAttrName(odsState.name), num_loops_val);
  }
}

void OrderedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars) {
  odsState.addOperands(depend_vec_vars);
  if (depend_type_val) {
    odsState.addAttribute(getDependTypeValAttrName(odsState.name), depend_type_val);
  }
  if (num_loops_val) {
    odsState.addAttribute(getNumLoopsValAttrName(odsState.name), num_loops_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrderedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OrderedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_depend_type_val;
  ::mlir::Attribute tblgen_num_loops_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getDependTypeValAttrName()) {
      tblgen_depend_type_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNumLoopsValAttrName()) {
      tblgen_num_loops_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_depend_type_val, "depend_type_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_num_loops_val, "num_loops_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OrderedOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OrderedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseDependAttr depend_type_valAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> depend_vec_varsOperands;
  ::llvm::SMLoc depend_vec_varsOperandsLoc;
  (void)depend_vec_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> depend_vec_varsTypes;
  if (::mlir::succeeded(parser.parseOptionalKeyword("depend_type"))) {

  if (parser.parseCustomAttributeWithFallback(depend_type_valAttr, ::mlir::Type{}, "depend_type_val",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("depend_vec"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  depend_vec_varsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(depend_vec_varsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(depend_vec_varsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(depend_vec_varsOperands, depend_vec_varsTypes, depend_vec_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OrderedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("depend_type_val")) {
    _odsPrinter << ' ' << "depend_type";
  _odsPrinter.printStrippedAttrOrType(getDependTypeValAttr());
  }
  if (!getDependVecVars().empty()) {
    _odsPrinter << ' ' << "depend_vec";
    _odsPrinter << "(";
    _odsPrinter << getDependVecVars();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getDependVecVars().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("depend_type_val");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
OrderedRegionOpGenericAdaptorBase::OrderedRegionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.ordered_region", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> OrderedRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr OrderedRegionOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr OrderedRegionOpGenericAdaptorBase::getSimdAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OrderedRegionOp::getSimdAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool OrderedRegionOpGenericAdaptorBase::getSimd() {
  auto attr = getSimdAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &OrderedRegionOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange OrderedRegionOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
OrderedRegionOpAdaptor::OrderedRegionOpAdaptor(OrderedRegionOp op) : OrderedRegionOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult OrderedRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_simd;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == OrderedRegionOp::getSimdAttrName(*odsOpName)) {
      tblgen_simd = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_simd && !((tblgen_simd.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.ordered_region' op ""attribute 'simd' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OrderedRegionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OrderedRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> OrderedRegionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrderedRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &OrderedRegionOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr OrderedRegionOp::getSimdAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSimdAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool OrderedRegionOp::getSimd() {
  auto attr = getSimdAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void OrderedRegionOp::setSimdAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getSimdAttrName(), attr);
}

void OrderedRegionOp::setSimd(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSimdAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getSimdAttrName());
}

::mlir::Attribute OrderedRegionOp::removeSimdAttr() {
  return (*this)->removeAttr(getSimdAttrName());
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr simd) {
  if (simd) {
    odsState.addAttribute(getSimdAttrName(odsState.name), simd);
  }
  (void)odsState.addRegion();
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr simd) {
  if (simd) {
    odsState.addAttribute(getSimdAttrName(odsState.name), simd);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool simd) {
  if (simd) {
    odsState.addAttribute(getSimdAttrName(odsState.name), ((simd) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool simd) {
  if (simd) {
    odsState.addAttribute(getSimdAttrName(odsState.name), ((simd) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrderedRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OrderedRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_simd;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSimdAttrName()) {
      tblgen_simd = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_simd, "simd")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult OrderedRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OrderedRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (::mlir::succeeded(parser.parseOptionalKeyword("simd"))) {
    result.addAttribute("simd", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void OrderedRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("simd")) {
    _odsPrinter << ' ' << "simd";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("simd");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSimdAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("simd");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedRegionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ParallelOpGenericAdaptorBase::ParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.parallel", odsAttrs.getContext());
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

::mlir::ArrayAttr ParallelOpGenericAdaptorBase::getReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::getReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > ParallelOpGenericAdaptorBase::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::omp::ClauseProcBindKindAttr ParallelOpGenericAdaptorBase::getProcBindValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::getProcBindValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseProcBindKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseProcBindKind> ParallelOpGenericAdaptorBase::getProcBindVal() {
  auto attr = getProcBindValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseProcBindKind>(attr.getValue()) : (::std::nullopt);
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
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.parallel' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ParallelOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_proc_bind_val;
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ParallelOp::getProcBindValAttrName(*odsOpName)) {
      tblgen_proc_bind_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::getReductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'omp.parallel' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.parallel' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_proc_bind_val && !((tblgen_proc_bind_val.isa<::mlir::omp::ClauseProcBindKindAttr>())))
    return emitError(loc, "'omp.parallel' op ""attribute 'proc_bind_val' failed to satisfy constraint: ProcBindKind Clause");
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

::mlir::TypedValue<::mlir::IntegerType> ParallelOp::getIfExprVar() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Value ParallelOp::getNumThreadsVar() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range ParallelOp::getAllocateVars() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ParallelOp::getAllocatorsVars() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ParallelOp::getReductionVars() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange ParallelOp::getIfExprVarMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getNumThreadsVarMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getAllocateVarsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getAllocatorsVarsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::getReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
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

::mlir::ArrayAttr ParallelOp::getReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > ParallelOp::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::omp::ClauseProcBindKindAttr ParallelOp::getProcBindValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getProcBindValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseProcBindKindAttr>();
}

::std::optional<::mlir::omp::ClauseProcBindKind> ParallelOp::getProcBindVal() {
  auto attr = getProcBindValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseProcBindKind>(attr.getValue()) : (::std::nullopt);
}

void ParallelOp::setReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReductionsAttrName(), attr);
}

void ParallelOp::setProcBindValAttr(::mlir::omp::ClauseProcBindKindAttr attr) {
  (*this)->setAttr(getProcBindValAttrName(), attr);
}

void ParallelOp::setProcBindVal(::std::optional<::mlir::omp::ClauseProcBindKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getProcBindValAttrName(), ::mlir::omp::ClauseProcBindKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getProcBindValAttrName());
}

::mlir::Attribute ParallelOp::removeReductionsAttr() {
  return (*this)->removeAttr(getReductionsAttrName());
}

::mlir::Attribute ParallelOp::removeProcBindValAttr() {
  return (*this)->removeAttr(getProcBindValAttrName());
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addOperands(reduction_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), static_cast<int32_t>(reduction_vars.size())}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (proc_bind_val) {
    odsState.addAttribute(getProcBindValAttrName(odsState.name), proc_bind_val);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addOperands(reduction_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), static_cast<int32_t>(reduction_vars.size())}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (proc_bind_val) {
    odsState.addAttribute(getProcBindValAttrName(odsState.name), proc_bind_val);
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
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_proc_bind_val;
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getProcBindValAttrName()) {
      tblgen_proc_bind_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getReductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps10(*this, tblgen_proc_bind_val, "proc_bind_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ParallelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ParallelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_expr_varOperands;
  ::llvm::SMLoc if_expr_varOperandsLoc;
  (void)if_expr_varOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> if_expr_varTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> num_threads_varOperands;
  ::llvm::SMLoc num_threads_varOperandsLoc;
  (void)num_threads_varOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> num_threads_varTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  ::mlir::omp::ClauseProcBindKindAttr proc_bind_valAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool reductionClause = false;
  bool ifClause = false;
  bool num_threadsClause = false;
  bool allocateClause = false;
  bool proc_bindClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
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
    if_expr_varOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_expr_varOperands.push_back(operand);
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
      if_expr_varTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_threads"))) {

  if (num_threadsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_threads` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_threadsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    num_threads_varOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      num_threads_varOperands.push_back(operand);
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
      num_threads_varTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("proc_bind"))) {

  if (proc_bindClause) {
    return parser.emitError(parser.getNameLoc())
          << "`proc_bind` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  proc_bindClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, proc_bind_valAttr))
      return ::mlir::failure();
    if (proc_bind_valAttr)
      result.addAttribute("proc_bind_val", proc_bind_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(if_expr_varOperands.size()), static_cast<int32_t>(num_threads_varOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size()), static_cast<int32_t>(reduction_varsOperands.size())}));
  if (parser.resolveOperands(if_expr_varOperands, if_expr_varTypes, if_expr_varOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(num_threads_varOperands, num_threads_varTypes, num_threads_varOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ParallelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getReductionVars().size() || getReductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getReductionVars(), getReductionVars().getTypes(), getReductionsAttr());
  _odsPrinter << ")";
  }
  if (false || getIfExprVar()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExprVar())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getIfExprVar() ? ::llvm::ArrayRef<::mlir::Type>(getIfExprVar().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNumThreadsVar()) {
  _odsPrinter << ' ' << "num_threads";
  _odsPrinter << "(";
  if (::mlir::Value value = getNumThreadsVar())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getNumThreadsVar() ? ::llvm::ArrayRef<::mlir::Type>(getNumThreadsVar().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getAllocateVars().size() || getAllocatorsVars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, getAllocateVars(), getAllocateVars().getTypes(), getAllocatorsVars(), getAllocatorsVars().getTypes());
  _odsPrinter << ")";
  }
  if (false || getProcBindValAttr()) {
  _odsPrinter << ' ' << "proc_bind";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getProcBindValAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("reductions");
  elidedAttrs.push_back("proc_bind_val");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReductionDeclareOpGenericAdaptorBase::ReductionDeclareOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.reduction.declare", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReductionDeclareOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReductionDeclareOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ReductionDeclareOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ReductionDeclareOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ReductionDeclareOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOpGenericAdaptorBase::getTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ReductionDeclareOp::getTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type ReductionDeclareOpGenericAdaptorBase::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::Region &ReductionDeclareOpGenericAdaptorBase::getInitializerRegion() {
  return *odsRegions[0];
}

::mlir::Region &ReductionDeclareOpGenericAdaptorBase::getReductionRegion() {
  return *odsRegions[1];
}

::mlir::Region &ReductionDeclareOpGenericAdaptorBase::getAtomicReductionRegion() {
  return *odsRegions[2];
}

::mlir::RegionRange ReductionDeclareOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(ReductionDeclareOp op) : ReductionDeclareOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReductionDeclareOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == ReductionDeclareOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'type'");
    if (namedAttrIt->getName() == ReductionDeclareOp::getTypeAttrName(*odsOpName)) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'omp.reduction.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'omp.reduction.declare' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReductionDeclareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReductionDeclareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ReductionDeclareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionDeclareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ReductionDeclareOp::getInitializerRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &ReductionDeclareOp::getReductionRegion() {
  return (*this)->getRegion(1);
}

::mlir::Region &ReductionDeclareOp::getAtomicReductionRegion() {
  return (*this)->getRegion(2);
}

::mlir::StringAttr ReductionDeclareOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ReductionDeclareOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOp::getTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type ReductionDeclareOp::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void ReductionDeclareOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void ReductionDeclareOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ReductionDeclareOp::setTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTypeAttrName(), attr);
}

void ReductionDeclareOp::setType(::mlir::Type attrValue) {
  (*this)->setAttr(getTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 3; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionDeclareOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'type'");
    if (namedAttrIt->getName() == getTypeAttrName()) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps4(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps2(*this, tblgen_type, "type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "initializerRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "reductionRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(2)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "atomicReductionRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReductionDeclareOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReductionDeclareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::TypeAttr typeAttr;
  std::unique_ptr<::mlir::Region> initializerRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> reductionRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> atomicReductionRegionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("init"))
    return ::mlir::failure();

  if (parser.parseRegion(*initializerRegionRegion))
    return ::mlir::failure();
  if (parser.parseKeyword("combiner"))
    return ::mlir::failure();

  if (parser.parseRegion(*reductionRegionRegion))
    return ::mlir::failure();
  {
    if (parseAtomicReductionRegion(parser, *atomicReductionRegionRegion))
      return ::mlir::failure();
  }
  result.addRegion(std::move(initializerRegionRegion));
  result.addRegion(std::move(reductionRegionRegion));
  result.addRegion(std::move(atomicReductionRegionRegion));
  return ::mlir::success();
}

void ReductionDeclareOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  elidedAttrs.push_back("type");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "init";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getInitializerRegion());
  _odsPrinter << ' ' << "combiner";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getReductionRegion());
  _odsPrinter << ' ';
  printAtomicReductionRegion(_odsPrinter, *this, getAtomicReductionRegion());
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReductionOpGenericAdaptorBase::ReductionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.reduction", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReductionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReductionOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReductionOpAdaptor::ReductionOpAdaptor(ReductionOp op) : ReductionOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReductionOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReductionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReductionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOp::getOperand() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> ReductionOp::getAccumulator() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ReductionOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReductionOp::getAccumulatorMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReductionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator) {
  odsState.addOperands(operand);
  odsState.addOperands(accumulator);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator) {
  odsState.addOperands(operand);
  odsState.addOperands(accumulator);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReductionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReductionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand accumulatorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accumulatorOperands(accumulatorRawOperands);  ::llvm::SMLoc accumulatorOperandsLoc;
  (void)accumulatorOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type accumulatorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accumulatorTypes(accumulatorRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  accumulatorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accumulatorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accumulatorRawTypes[0] = type;
  }
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accumulatorOperands, accumulatorTypes, accumulatorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReductionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getAccumulator();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getAccumulator().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SectionOpGenericAdaptorBase::SectionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.section", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SectionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SectionOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &SectionOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange SectionOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
SectionOpAdaptor::SectionOpAdaptor(SectionOp op) : SectionOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SectionOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SectionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SectionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SectionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SectionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SectionOp::getRegion() {
  return (*this)->getRegion(0);
}

void SectionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
  (void)odsState.addRegion();
}

void SectionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SectionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SectionOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SectionOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SectionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void SectionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SectionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SectionsOpGenericAdaptorBase::SectionsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.sections", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SectionsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SectionsOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SectionsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SectionsOpGenericAdaptorBase::getReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SectionsOp::getReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > SectionsOpGenericAdaptorBase::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::UnitAttr SectionsOpGenericAdaptorBase::getNowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SectionsOp::getNowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool SectionsOpGenericAdaptorBase::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &SectionsOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange SectionsOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
SectionsOpAdaptor::SectionsOpAdaptor(SectionsOp op) : SectionsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SectionsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.sections' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SectionsOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SectionsOp::getNowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SectionsOp::getReductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'omp.sections' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.sections' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.sections' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SectionsOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SectionsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SectionsOp::getReductionVars() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SectionsOp::getAllocateVars() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SectionsOp::getAllocatorsVars() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SectionsOp::getReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SectionsOp::getAllocateVarsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SectionsOp::getAllocatorsVarsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SectionsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SectionsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SectionsOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr SectionsOp::getReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > SectionsOp::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::UnitAttr SectionsOp::getNowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SectionsOp::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void SectionsOp::setReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReductionsAttrName(), attr);
}

void SectionsOp::setNowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNowaitAttrName(), attr);
}

void SectionsOp::setNowait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNowaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNowaitAttrName());
}

::mlir::Attribute SectionsOp::removeReductionsAttr() {
  return (*this)->removeAttr(getReductionsAttrName());
}

::mlir::Attribute SectionsOp::removeNowaitAttr() {
  return (*this)->removeAttr(getNowaitAttrName());
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SectionsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SectionsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SectionsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SectionsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool reductionClause = false;
  bool allocateClause = false;
  bool nowaitClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(reduction_varsOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SectionsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getReductionVars().size() || getReductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getReductionVars(), getReductionVars().getTypes(), getReductionsAttr());
  _odsPrinter << ")";
  }
  if (false || getAllocateVars().size() || getAllocatorsVars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, getAllocateVars(), getAllocateVars().getTypes(), getAllocatorsVars(), getAllocatorsVars().getTypes());
  _odsPrinter << ")";
  }
  if (false || getNowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("reductions");
  elidedAttrs.push_back("nowait");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNowaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nowait");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SectionsOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SimdLoopOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SimdLoopOpGenericAdaptorBase::SimdLoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.simdloop", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SimdLoopOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SimdLoopOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SimdLoopOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SimdLoopOpGenericAdaptorBase::getAlignmentValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SimdLoopOp::getAlignmentValuesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > SimdLoopOpGenericAdaptorBase::getAlignmentValues() {
  auto attr = getAlignmentValuesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::omp::ClauseOrderKindAttr SimdLoopOpGenericAdaptorBase::getOrderValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SimdLoopOp::getOrderValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseOrderKind> SimdLoopOpGenericAdaptorBase::getOrderVal() {
  auto attr = getOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr SimdLoopOpGenericAdaptorBase::getSimdlenAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SimdLoopOp::getSimdlenAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> SimdLoopOpGenericAdaptorBase::getSimdlen() {
  auto attr = getSimdlenAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::IntegerAttr SimdLoopOpGenericAdaptorBase::getSafelenAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SimdLoopOp::getSafelenAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> SimdLoopOpGenericAdaptorBase::getSafelen() {
  auto attr = getSafelenAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr SimdLoopOpGenericAdaptorBase::getInclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SimdLoopOp::getInclusiveAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool SimdLoopOpGenericAdaptorBase::getInclusive() {
  auto attr = getInclusiveAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &SimdLoopOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange SimdLoopOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
SimdLoopOpAdaptor::SimdLoopOpAdaptor(SimdLoopOp op) : SimdLoopOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SimdLoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_alignment_values;
  ::mlir::Attribute tblgen_inclusive;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.simdloop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SimdLoopOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SimdLoopOp::getAlignmentValuesAttrName(*odsOpName)) {
      tblgen_alignment_values = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SimdLoopOp::getInclusiveAttrName(*odsOpName)) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_order_val;
  ::mlir::Attribute tblgen_safelen;
  ::mlir::Attribute tblgen_simdlen;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SimdLoopOp::getOrderValAttrName(*odsOpName)) {
      tblgen_order_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SimdLoopOp::getSafelenAttrName(*odsOpName)) {
      tblgen_safelen = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SimdLoopOp::getSimdlenAttrName(*odsOpName)) {
      tblgen_simdlen = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 6)
      return emitError(loc, "'omp.simdloop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 6 "
                "elements, but got ") << numElements;
  }

  if (tblgen_alignment_values && !(((tblgen_alignment_values.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alignment_values.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'omp.simdloop' op ""attribute 'alignment_values' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_order_val && !((tblgen_order_val.isa<::mlir::omp::ClauseOrderKindAttr>())))
    return emitError(loc, "'omp.simdloop' op ""attribute 'order_val' failed to satisfy constraint: OrderKind Clause");

  if (tblgen_simdlen && !((((tblgen_simdlen.isa<::mlir::IntegerAttr>())) && ((tblgen_simdlen.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_simdlen.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'omp.simdloop' op ""attribute 'simdlen' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive");

  if (tblgen_safelen && !((((tblgen_safelen.isa<::mlir::IntegerAttr>())) && ((tblgen_safelen.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_safelen.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'omp.simdloop' op ""attribute 'safelen' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive");

  if (tblgen_inclusive && !((tblgen_inclusive.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.simdloop' op ""attribute 'inclusive' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SimdLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SimdLoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SimdLoopOp::getLowerBound() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SimdLoopOp::getUpperBound() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SimdLoopOp::getStep() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range SimdLoopOp::getAlignedVars() {
  return getODSOperands(3);
}

::mlir::TypedValue<::mlir::IntegerType> SimdLoopOp::getIfExpr() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range SimdLoopOp::getNontemporalVars() {
  return getODSOperands(5);
}

::mlir::MutableOperandRange SimdLoopOp::getLowerBoundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::getUpperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::getStepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::getAlignedVarsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::getNontemporalVarsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SimdLoopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SimdLoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SimdLoopOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr SimdLoopOp::getAlignmentValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAlignmentValuesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > SimdLoopOp::getAlignmentValues() {
  auto attr = getAlignmentValuesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::omp::ClauseOrderKindAttr SimdLoopOp::getOrderValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOrderValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
}

::std::optional<::mlir::omp::ClauseOrderKind> SimdLoopOp::getOrderVal() {
  auto attr = getOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr SimdLoopOp::getSimdlenAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSimdlenAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> SimdLoopOp::getSimdlen() {
  auto attr = getSimdlenAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::IntegerAttr SimdLoopOp::getSafelenAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSafelenAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> SimdLoopOp::getSafelen() {
  auto attr = getSafelenAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr SimdLoopOp::getInclusiveAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInclusiveAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SimdLoopOp::getInclusive() {
  auto attr = getInclusiveAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void SimdLoopOp::setAlignmentValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAlignmentValuesAttrName(), attr);
}

void SimdLoopOp::setOrderValAttr(::mlir::omp::ClauseOrderKindAttr attr) {
  (*this)->setAttr(getOrderValAttrName(), attr);
}

void SimdLoopOp::setOrderVal(::std::optional<::mlir::omp::ClauseOrderKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOrderValAttrName(), ::mlir::omp::ClauseOrderKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getOrderValAttrName());
}

void SimdLoopOp::setSimdlenAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getSimdlenAttrName(), attr);
}

void SimdLoopOp::setSimdlen(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSimdlenAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getSimdlenAttrName());
}

void SimdLoopOp::setSafelenAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getSafelenAttrName(), attr);
}

void SimdLoopOp::setSafelen(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSafelenAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getSafelenAttrName());
}

void SimdLoopOp::setInclusiveAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInclusiveAttrName(), attr);
}

void SimdLoopOp::setInclusive(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInclusiveAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInclusiveAttrName());
}

::mlir::Attribute SimdLoopOp::removeAlignmentValuesAttr() {
  return (*this)->removeAttr(getAlignmentValuesAttrName());
}

::mlir::Attribute SimdLoopOp::removeOrderValAttr() {
  return (*this)->removeAttr(getOrderValAttrName());
}

::mlir::Attribute SimdLoopOp::removeSimdlenAttr() {
  return (*this)->removeAttr(getSimdlenAttrName());
}

::mlir::Attribute SimdLoopOp::removeSafelenAttr() {
  return (*this)->removeAttr(getSafelenAttrName());
}

::mlir::Attribute SimdLoopOp::removeInclusiveAttr() {
  return (*this)->removeAttr(getInclusiveAttrName());
}

void SimdLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/::mlir::UnitAttr inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(aligned_vars);
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addOperands(nontemporal_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(aligned_vars.size()), (if_expr ? 1 : 0), static_cast<int32_t>(nontemporal_vars.size())}));
  if (alignment_values) {
    odsState.addAttribute(getAlignmentValuesAttrName(odsState.name), alignment_values);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (simdlen) {
    odsState.addAttribute(getSimdlenAttrName(odsState.name), simdlen);
  }
  if (safelen) {
    odsState.addAttribute(getSafelenAttrName(odsState.name), safelen);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  }
  (void)odsState.addRegion();
}

void SimdLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/::mlir::UnitAttr inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(aligned_vars);
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addOperands(nontemporal_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(aligned_vars.size()), (if_expr ? 1 : 0), static_cast<int32_t>(nontemporal_vars.size())}));
  if (alignment_values) {
    odsState.addAttribute(getAlignmentValuesAttrName(odsState.name), alignment_values);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (simdlen) {
    odsState.addAttribute(getSimdlenAttrName(odsState.name), simdlen);
  }
  if (safelen) {
    odsState.addAttribute(getSafelenAttrName(odsState.name), safelen);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SimdLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/bool inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(aligned_vars);
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addOperands(nontemporal_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(aligned_vars.size()), (if_expr ? 1 : 0), static_cast<int32_t>(nontemporal_vars.size())}));
  if (alignment_values) {
    odsState.addAttribute(getAlignmentValuesAttrName(odsState.name), alignment_values);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (simdlen) {
    odsState.addAttribute(getSimdlenAttrName(odsState.name), simdlen);
  }
  if (safelen) {
    odsState.addAttribute(getSafelenAttrName(odsState.name), safelen);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), ((inclusive) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
}

void SimdLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange aligned_vars, /*optional*/::mlir::ArrayAttr alignment_values, /*optional*/::mlir::Value if_expr, ::mlir::ValueRange nontemporal_vars, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/bool inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(aligned_vars);
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addOperands(nontemporal_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(aligned_vars.size()), (if_expr ? 1 : 0), static_cast<int32_t>(nontemporal_vars.size())}));
  if (alignment_values) {
    odsState.addAttribute(getAlignmentValuesAttrName(odsState.name), alignment_values);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (simdlen) {
    odsState.addAttribute(getSimdlenAttrName(odsState.name), simdlen);
  }
  if (safelen) {
    odsState.addAttribute(getSafelenAttrName(odsState.name), safelen);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), ((inclusive) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SimdLoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SimdLoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_alignment_values;
  ::mlir::Attribute tblgen_inclusive;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentValuesAttrName()) {
      tblgen_alignment_values = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getInclusiveAttrName()) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_order_val;
  ::mlir::Attribute tblgen_safelen;
  ::mlir::Attribute tblgen_simdlen;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getOrderValAttrName()) {
      tblgen_order_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSafelenAttrName()) {
      tblgen_safelen = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSimdlenAttrName()) {
      tblgen_simdlen = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 6)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 6 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_alignment_values, "alignment_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps12(*this, tblgen_order_val, "order_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps13(*this, tblgen_simdlen, "simdlen")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps13(*this, tblgen_safelen, "safelen")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_inclusive, "inclusive")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((this->getODSOperands(0).getType()) == (this->getODSOperands(1).getType()) && (this->getODSOperands(1).getType()) == (this->getODSOperands(2).getType()) && (this->getODSOperands(2).getType()) == (this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that all of {lowerBound, upperBound, step} have same type");
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SimdLoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SimdLoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> aligned_varsOperands;
  ::llvm::SMLoc aligned_varsOperandsLoc;
  (void)aligned_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> aligned_varsTypes;
  ::mlir::ArrayAttr alignment_valuesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> nontemporal_varsOperands;
  ::llvm::SMLoc nontemporal_varsOperandsLoc;
  (void)nontemporal_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> nontemporal_varsTypes;
  ::mlir::omp::ClauseOrderKindAttr order_valAttr;
  ::mlir::IntegerAttr simdlenAttr;
  ::mlir::IntegerAttr safelenAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lowerBoundOperands;
  ::llvm::SMLoc lowerBoundOperandsLoc;
  (void)lowerBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> upperBoundOperands;
  ::llvm::SMLoc upperBoundOperandsLoc;
  (void)upperBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stepOperands;
  ::llvm::SMLoc stepOperandsLoc;
  (void)stepOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> stepTypes;
  ::mlir::UnitAttr inclusiveAttr;
  bool alignedClause = false;
  bool ifClause = false;
  bool nontemporalClause = false;
  bool orderClause = false;
  bool simdlenClause = false;
  bool safelenClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("aligned"))) {

  if (alignedClause) {
    return parser.emitError(parser.getNameLoc())
          << "`aligned` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  alignedClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    aligned_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAlignedClause(parser, aligned_varsOperands, aligned_varsTypes, alignment_valuesAttr))
      return ::mlir::failure();
    if (alignment_valuesAttr)
      result.addAttribute("alignment_values", alignment_valuesAttr);
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nontemporal"))) {

  if (nontemporalClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nontemporal` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nontemporalClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  nontemporal_varsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(nontemporal_varsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(nontemporal_varsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("order"))) {

  if (orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, order_valAttr))
      return ::mlir::failure();
    if (order_valAttr)
      result.addAttribute("order_val", order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("simdlen"))) {

  if (simdlenClause) {
    return parser.emitError(parser.getNameLoc())
          << "`simdlen` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  simdlenClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(simdlenAttr, parser.getBuilder().getIntegerType(64), "simdlen", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("safelen"))) {

  if (safelenClause) {
    return parser.emitError(parser.getNameLoc())
          << "`safelen` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  safelenClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(safelenAttr, parser.getBuilder().getIntegerType(64), "safelen", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseKeyword("for"))
    return ::mlir::failure();
  {
    lowerBoundOperandsLoc = parser.getCurrentLocation();
    upperBoundOperandsLoc = parser.getCurrentLocation();
    stepOperandsLoc = parser.getCurrentLocation();
    if (parseLoopControl(parser, *regionRegion, lowerBoundOperands, upperBoundOperands, stepOperands, stepTypes, inclusiveAttr))
      return ::mlir::failure();
    if (inclusiveAttr)
      result.addAttribute("inclusive", inclusiveAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(lowerBoundOperands.size()), static_cast<int32_t>(upperBoundOperands.size()), static_cast<int32_t>(stepOperands.size()), static_cast<int32_t>(aligned_varsOperands.size()), static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(nontemporal_varsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(lowerBoundOperands, stepTypes, lowerBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperBoundOperands, stepTypes, upperBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stepOperands, stepTypes, stepOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aligned_varsOperands, aligned_varsTypes, aligned_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(nontemporal_varsOperands, nontemporal_varsTypes, nontemporal_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SimdLoopOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getAlignedVars().size() || getAlignmentValuesAttr()) {
  _odsPrinter << ' ' << "aligned";
  _odsPrinter << "(";
  printAlignedClause(_odsPrinter, *this, getAlignedVars(), getAlignedVars().getTypes(), getAlignmentValuesAttr());
  _odsPrinter << ")";
  }
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getNontemporalVars().size()) {
  _odsPrinter << ' ' << "nontemporal";
  _odsPrinter << "(";
  _odsPrinter << getNontemporalVars();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getNontemporalVars().getTypes();
  _odsPrinter << ")";
  }
  if (false || getOrderValAttr()) {
  _odsPrinter << ' ' << "order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getOrderValAttr());
  _odsPrinter << ")";
  }
  if (false || getSimdlenAttr()) {
  _odsPrinter << ' ' << "simdlen";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getSimdlenAttr());
  _odsPrinter << ")";
  }
  if (false || getSafelenAttr()) {
  _odsPrinter << ' ' << "safelen";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getSafelenAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "for";
  _odsPrinter << ' ';
  printLoopControl(_odsPrinter, *this, getRegion(), getLowerBound(), getUpperBound(), getStep(), getStep().getTypes(), getInclusiveAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("alignment_values");
  elidedAttrs.push_back("order_val");
  elidedAttrs.push_back("simdlen");
  elidedAttrs.push_back("safelen");
  elidedAttrs.push_back("inclusive");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInclusiveAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inclusive");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SimdLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SingleOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SingleOpGenericAdaptorBase::SingleOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.single", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SingleOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SingleOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SingleOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr SingleOpGenericAdaptorBase::getNowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SingleOp::getNowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool SingleOpGenericAdaptorBase::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &SingleOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange SingleOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
SingleOpAdaptor::SingleOpAdaptor(SingleOp op) : SingleOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SingleOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.single' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SingleOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SingleOp::getNowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'omp.single' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.single' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SingleOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SingleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SingleOp::getAllocateVars() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SingleOp::getAllocatorsVars() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange SingleOp::getAllocateVarsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SingleOp::getAllocatorsVarsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SingleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SingleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SingleOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr SingleOp::getNowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SingleOp::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void SingleOp::setNowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNowaitAttrName(), attr);
}

void SingleOp::setNowait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNowaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNowaitAttrName());
}

::mlir::Attribute SingleOp::removeNowaitAttr() {
  return (*this)->removeAttr(getNowaitAttrName());
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SingleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SingleOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SingleOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SingleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool allocateClause = false;
  bool nowaitClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SingleOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getAllocateVars().size() || getAllocatorsVars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, getAllocateVars(), getAllocateVars().getTypes(), getAllocatorsVars(), getAllocatorsVars().getTypes());
  _odsPrinter << ")";
  }
  if (false || getNowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("nowait");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNowaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nowait");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SingleOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TargetOpGenericAdaptorBase::TargetOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.target", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TargetOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TargetOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr TargetOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TargetOpGenericAdaptorBase::getNowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TargetOp::getNowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TargetOpGenericAdaptorBase::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr TargetOpGenericAdaptorBase::getMapTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TargetOp::getMapTypesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TargetOpGenericAdaptorBase::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::Region &TargetOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange TargetOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
TargetOpAdaptor::TargetOpAdaptor(TargetOp op) : TargetOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TargetOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_map_types;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TargetOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TargetOp::getMapTypesAttrName(*odsOpName)) {
      tblgen_map_types = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TargetOp::getNowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'omp.target' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.target' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");

  if (tblgen_map_types && !(((tblgen_map_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_map_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'omp.target' op ""attribute 'map_types' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TargetOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range TargetOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> TargetOp::getIfExpr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> TargetOp::getDevice() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> TargetOp::getThreadLimit() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range TargetOp::getMapOperands() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange TargetOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::getDeviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::getThreadLimitMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TargetOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TargetOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TargetOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr TargetOp::getNowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TargetOp::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr TargetOp::getMapTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMapTypesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TargetOp::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void TargetOp::setNowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNowaitAttrName(), attr);
}

void TargetOp::setNowait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNowaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNowaitAttrName());
}

void TargetOp::setMapTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMapTypesAttrName(), attr);
}

::mlir::Attribute TargetOp::removeNowaitAttr() {
  return (*this)->removeAttr(getNowaitAttrName());
}

::mlir::Attribute TargetOp::removeMapTypesAttr() {
  return (*this)->removeAttr(getMapTypesAttrName());
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  if (map_types) {
    odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  }
  (void)odsState.addRegion();
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  if (map_types) {
    odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (map_types) {
    odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  }
  (void)odsState.addRegion();
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait, ::mlir::ValueRange map_operands, /*optional*/::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (map_types) {
    odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TargetOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TargetOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_map_types;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getMapTypesAttrName()) {
      tblgen_map_types = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_map_types, "map_types")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TargetOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TargetOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperands;
  ::llvm::SMLoc deviceOperandsLoc;
  (void)deviceOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> thread_limitOperands;
  ::llvm::SMLoc thread_limitOperandsLoc;
  (void)thread_limitOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> thread_limitTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> map_operandsOperands;
  ::llvm::SMLoc map_operandsOperandsLoc;
  (void)map_operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> map_operandsTypes;
  ::mlir::ArrayAttr map_typesAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool deviceClause = false;
  bool thread_limitClause = false;
  bool nowaitClause = false;
  bool mapClause = false;
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
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

  {
    deviceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceOperands.push_back(operand);
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
      deviceTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("thread_limit"))) {

  if (thread_limitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`thread_limit` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  thread_limitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    thread_limitOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      thread_limitOperands.push_back(operand);
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
      thread_limitTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("map"))) {

  if (mapClause) {
    return parser.emitError(parser.getNameLoc())
          << "`map` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  mapClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    map_operandsOperandsLoc = parser.getCurrentLocation();
    if (parseMapClause(parser, map_operandsOperands, map_operandsTypes, map_typesAttr))
      return ::mlir::failure();
    if (map_typesAttr)
      result.addAttribute("map_types", map_typesAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(deviceOperands.size()), static_cast<int32_t>(thread_limitOperands.size()), static_cast<int32_t>(map_operandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperands, deviceTypes, deviceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(thread_limitOperands, thread_limitTypes, thread_limitOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(map_operandsOperands, map_operandsTypes, map_operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TargetOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getDevice()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  if (::mlir::Value value = getDevice())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getDevice() ? ::llvm::ArrayRef<::mlir::Type>(getDevice().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getThreadLimit()) {
  _odsPrinter << ' ' << "thread_limit";
  _odsPrinter << "(";
  if (::mlir::Value value = getThreadLimit())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getThreadLimit() ? ::llvm::ArrayRef<::mlir::Type>(getThreadLimit().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  if (false || getMapOperands().size() || getMapTypesAttr()) {
  _odsPrinter << ' ' << "map";
  _odsPrinter << "(";
  printMapClause(_odsPrinter, *this, getMapOperands(), getMapOperands().getTypes(), getMapTypesAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("nowait");
  elidedAttrs.push_back("map_types");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNowaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nowait");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::DataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DataOpGenericAdaptorBase::DataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.target_data", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DataOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr DataOpGenericAdaptorBase::getMapTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DataOp::getMapTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DataOpGenericAdaptorBase::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr;
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
  ::mlir::Attribute tblgen_map_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target_data' op ""requires attribute 'map_types'");
    if (namedAttrIt->getName() == DataOp::getMapTypesAttrName(*odsOpName)) {
      tblgen_map_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DataOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'omp.target_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_map_types && !(((tblgen_map_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_map_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'omp.target_data' op ""attribute 'map_types' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

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

::mlir::TypedValue<::mlir::IntegerType> DataOp::getIfExpr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> DataOp::getDevice() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range DataOp::getUseDevicePtr() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range DataOp::getUseDeviceAddr() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range DataOp::getMapOperands() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange DataOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getDeviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getUseDevicePtrMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getUseDeviceAddrMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
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

::mlir::ArrayAttr DataOp::getMapTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMapTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DataOp::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr;
}

void DataOp::setMapTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMapTypesAttrName(), attr);
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, ::mlir::ValueRange use_device_ptr, ::mlir::ValueRange use_device_addr, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(use_device_ptr);
  odsState.addOperands(use_device_addr);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(use_device_ptr.size()), static_cast<int32_t>(use_device_addr.size()), static_cast<int32_t>(map_operands.size())}));
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  (void)odsState.addRegion();
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, ::mlir::ValueRange use_device_ptr, ::mlir::ValueRange use_device_addr, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(use_device_ptr);
  odsState.addOperands(use_device_addr);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(use_device_ptr.size()), static_cast<int32_t>(use_device_addr.size()), static_cast<int32_t>(map_operands.size())}));
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
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
  ::mlir::Attribute tblgen_map_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'map_types'");
    if (namedAttrIt->getName() == getMapTypesAttrName()) {
      tblgen_map_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
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
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_map_types, "map_types")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> if_exprTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperands;
  ::llvm::SMLoc deviceOperandsLoc;
  (void)deviceOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> use_device_ptrOperands;
  ::llvm::SMLoc use_device_ptrOperandsLoc;
  (void)use_device_ptrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> use_device_ptrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> use_device_addrOperands;
  ::llvm::SMLoc use_device_addrOperandsLoc;
  (void)use_device_addrOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> use_device_addrTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> map_operandsOperands;
  ::llvm::SMLoc map_operandsOperandsLoc;
  (void)map_operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> map_operandsTypes;
  ::mlir::ArrayAttr map_typesAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool deviceClause = false;
  bool use_device_ptrClause = false;
  bool use_device_addrClause = false;
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
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
      if_exprTypes.push_back(optionalType);
    }
  }
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

  {
    deviceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceOperands.push_back(operand);
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
      deviceTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("use_device_ptr"))) {

  if (use_device_ptrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`use_device_ptr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  use_device_ptrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  use_device_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(use_device_ptrOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(use_device_ptrTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("use_device_addr"))) {

  if (use_device_addrClause) {
    return parser.emitError(parser.getNameLoc())
          << "`use_device_addr` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  use_device_addrClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  use_device_addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(use_device_addrOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(use_device_addrTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseKeyword("map"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    map_operandsOperandsLoc = parser.getCurrentLocation();
    if (parseMapClause(parser, map_operandsOperands, map_operandsTypes, map_typesAttr))
      return ::mlir::failure();
    result.addAttribute("map_types", map_typesAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(deviceOperands.size()), static_cast<int32_t>(use_device_ptrOperands.size()), static_cast<int32_t>(use_device_addrOperands.size()), static_cast<int32_t>(map_operandsOperands.size())}));
  if (parser.resolveOperands(if_exprOperands, if_exprTypes, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperands, deviceTypes, deviceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(use_device_ptrOperands, use_device_ptrTypes, use_device_ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(use_device_addrOperands, use_device_addrTypes, use_device_addrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(map_operandsOperands, map_operandsTypes, map_operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getIfExpr() ? ::llvm::ArrayRef<::mlir::Type>(getIfExpr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getDevice()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  if (::mlir::Value value = getDevice())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getDevice() ? ::llvm::ArrayRef<::mlir::Type>(getDevice().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getUseDevicePtr().size()) {
  _odsPrinter << ' ' << "use_device_ptr";
  _odsPrinter << "(";
  _odsPrinter << getUseDevicePtr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getUseDevicePtr().getTypes();
  _odsPrinter << ")";
  }
  if (false || getUseDeviceAddr().size()) {
  _odsPrinter << ' ' << "use_device_addr";
  _odsPrinter << "(";
  _odsPrinter << getUseDeviceAddr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getUseDeviceAddr().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "map";
  _odsPrinter << "(";
  printMapClause(_odsPrinter, *this, getMapOperands(), getMapOperands().getTypes(), getMapTypesAttr());
  _odsPrinter << ")";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("map_types");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::DataOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::EnterDataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EnterDataOpGenericAdaptorBase::EnterDataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.target_enter_data", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EnterDataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, EnterDataOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr EnterDataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr EnterDataOpGenericAdaptorBase::getNowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, EnterDataOp::getNowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool EnterDataOpGenericAdaptorBase::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr EnterDataOpGenericAdaptorBase::getMapTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, EnterDataOp::getMapTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr EnterDataOpGenericAdaptorBase::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr;
}

} // namespace detail
EnterDataOpAdaptor::EnterDataOpAdaptor(EnterDataOp op) : EnterDataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult EnterDataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target_enter_data' op ""requires attribute 'map_types'");
    if (namedAttrIt->getName() == EnterDataOp::getMapTypesAttrName(*odsOpName)) {
      tblgen_map_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target_enter_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == EnterDataOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == EnterDataOp::getNowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'omp.target_enter_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.target_enter_data' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");

  if (tblgen_map_types && !(((tblgen_map_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_map_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'omp.target_enter_data' op ""attribute 'map_types' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EnterDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

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

::mlir::TypedValue<::mlir::IntegerType> EnterDataOp::getIfExpr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> EnterDataOp::getDevice() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range EnterDataOp::getMapOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange EnterDataOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getDeviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
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

::mlir::UnitAttr EnterDataOp::getNowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getNowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr EnterDataOp::getMapTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMapTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr EnterDataOp::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr;
}

void EnterDataOp::setNowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNowaitAttrName(), attr);
}

void EnterDataOp::setNowait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNowaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNowaitAttrName());
}

void EnterDataOp::setMapTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMapTypesAttrName(), attr);
}

::mlir::Attribute EnterDataOp::removeNowaitAttr() {
  return (*this)->removeAttr(getNowaitAttrName());
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
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
  ::mlir::Attribute tblgen_map_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'map_types'");
    if (namedAttrIt->getName() == getMapTypesAttrName()) {
      tblgen_map_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_map_types, "map_types")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> if_exprTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperands;
  ::llvm::SMLoc deviceOperandsLoc;
  (void)deviceOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> map_operandsOperands;
  ::llvm::SMLoc map_operandsOperandsLoc;
  (void)map_operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> map_operandsTypes;
  ::mlir::ArrayAttr map_typesAttr;
  bool ifClause = false;
  bool deviceClause = false;
  bool nowaitClause = false;
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
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
      if_exprTypes.push_back(optionalType);
    }
  }
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

  {
    deviceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceOperands.push_back(operand);
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
      deviceTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}
  if (parser.parseKeyword("map"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    map_operandsOperandsLoc = parser.getCurrentLocation();
    if (parseMapClause(parser, map_operandsOperands, map_operandsTypes, map_typesAttr))
      return ::mlir::failure();
    result.addAttribute("map_types", map_typesAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(deviceOperands.size()), static_cast<int32_t>(map_operandsOperands.size())}));
  if (parser.resolveOperands(if_exprOperands, if_exprTypes, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperands, deviceTypes, deviceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(map_operandsOperands, map_operandsTypes, map_operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EnterDataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getIfExpr() ? ::llvm::ArrayRef<::mlir::Type>(getIfExpr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getDevice()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  if (::mlir::Value value = getDevice())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getDevice() ? ::llvm::ArrayRef<::mlir::Type>(getDevice().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ' << "map";
  _odsPrinter << "(";
  printMapClause(_odsPrinter, *this, getMapOperands(), getMapOperands().getTypes(), getMapTypesAttr());
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("nowait");
  elidedAttrs.push_back("map_types");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNowaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nowait");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::EnterDataOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ExitDataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExitDataOpGenericAdaptorBase::ExitDataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.target_exit_data", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExitDataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ExitDataOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ExitDataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ExitDataOpGenericAdaptorBase::getNowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, ExitDataOp::getNowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ExitDataOpGenericAdaptorBase::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr ExitDataOpGenericAdaptorBase::getMapTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ExitDataOp::getMapTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExitDataOpGenericAdaptorBase::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr;
}

} // namespace detail
ExitDataOpAdaptor::ExitDataOpAdaptor(ExitDataOp op) : ExitDataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExitDataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target_exit_data' op ""requires attribute 'map_types'");
    if (namedAttrIt->getName() == ExitDataOp::getMapTypesAttrName(*odsOpName)) {
      tblgen_map_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target_exit_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ExitDataOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ExitDataOp::getNowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'omp.target_exit_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.target_exit_data' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");

  if (tblgen_map_types && !(((tblgen_map_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_map_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'omp.target_exit_data' op ""attribute 'map_types' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExitDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

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

::mlir::TypedValue<::mlir::IntegerType> ExitDataOp::getIfExpr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> ExitDataOp::getDevice() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range ExitDataOp::getMapOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ExitDataOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getDeviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
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

::mlir::UnitAttr ExitDataOp::getNowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getNowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr ExitDataOp::getMapTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMapTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExitDataOp::getMapTypes() {
  auto attr = getMapTypesAttr();
  return attr;
}

void ExitDataOp::setNowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNowaitAttrName(), attr);
}

void ExitDataOp::setNowait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNowaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNowaitAttrName());
}

void ExitDataOp::setMapTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMapTypesAttrName(), attr);
}

::mlir::Attribute ExitDataOp::removeNowaitAttr() {
  return (*this)->removeAttr(getNowaitAttrName());
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::UnitAttr nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/bool nowait, ::mlir::ValueRange map_operands, ::mlir::ArrayAttr map_types) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  odsState.addOperands(map_operands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (device ? 1 : 0), static_cast<int32_t>(map_operands.size())}));
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getMapTypesAttrName(odsState.name), map_types);
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
  ::mlir::Attribute tblgen_map_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'map_types'");
    if (namedAttrIt->getName() == getMapTypesAttrName()) {
      tblgen_map_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_map_types, "map_types")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> if_exprTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperands;
  ::llvm::SMLoc deviceOperandsLoc;
  (void)deviceOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> map_operandsOperands;
  ::llvm::SMLoc map_operandsOperandsLoc;
  (void)map_operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> map_operandsTypes;
  ::mlir::ArrayAttr map_typesAttr;
  bool ifClause = false;
  bool deviceClause = false;
  bool nowaitClause = false;
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
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
      if_exprTypes.push_back(optionalType);
    }
  }
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

  {
    deviceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceOperands.push_back(operand);
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
      deviceTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}
  if (parser.parseKeyword("map"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    map_operandsOperandsLoc = parser.getCurrentLocation();
    if (parseMapClause(parser, map_operandsOperands, map_operandsTypes, map_typesAttr))
      return ::mlir::failure();
    result.addAttribute("map_types", map_typesAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(deviceOperands.size()), static_cast<int32_t>(map_operandsOperands.size())}));
  if (parser.resolveOperands(if_exprOperands, if_exprTypes, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperands, deviceTypes, deviceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(map_operandsOperands, map_operandsTypes, map_operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExitDataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getIfExpr() ? ::llvm::ArrayRef<::mlir::Type>(getIfExpr().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getDevice()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  if (::mlir::Value value = getDevice())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getDevice() ? ::llvm::ArrayRef<::mlir::Type>(getDevice().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ' << "map";
  _odsPrinter << "(";
  printMapClause(_odsPrinter, *this, getMapOperands(), getMapOperands().getTypes(), getMapTypesAttr());
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("nowait");
  elidedAttrs.push_back("map_types");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNowaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nowait");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ExitDataOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskGroupOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TaskGroupOpGenericAdaptorBase::TaskGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.taskgroup", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TaskGroupOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TaskGroupOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr TaskGroupOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TaskGroupOpGenericAdaptorBase::getTaskReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TaskGroupOp::getTaskReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TaskGroupOpGenericAdaptorBase::getTaskReductions() {
  auto attr = getTaskReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::Region &TaskGroupOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange TaskGroupOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
TaskGroupOpAdaptor::TaskGroupOpAdaptor(TaskGroupOp op) : TaskGroupOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TaskGroupOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.taskgroup' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TaskGroupOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_task_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TaskGroupOp::getTaskReductionsAttrName(*odsOpName)) {
      tblgen_task_reductions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'omp.taskgroup' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_task_reductions && !(((tblgen_task_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_task_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.taskgroup' op ""attribute 'task_reductions' failed to satisfy constraint: symbol ref array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskGroupOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range TaskGroupOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range TaskGroupOp::getTaskReductionVars() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range TaskGroupOp::getAllocateVars() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range TaskGroupOp::getAllocatorsVars() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange TaskGroupOp::getTaskReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskGroupOp::getAllocateVarsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskGroupOp::getAllocatorsVarsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TaskGroupOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskGroupOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TaskGroupOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr TaskGroupOp::getTaskReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTaskReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TaskGroupOp::getTaskReductions() {
  auto attr = getTaskReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void TaskGroupOp::setTaskReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTaskReductionsAttrName(), attr);
}

::mlir::Attribute TaskGroupOp::removeTaskReductionsAttr() {
  return (*this)->removeAttr(getTaskReductionsAttrName());
}

void TaskGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange task_reduction_vars, /*optional*/::mlir::ArrayAttr task_reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  odsState.addOperands(task_reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(task_reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (task_reductions) {
    odsState.addAttribute(getTaskReductionsAttrName(odsState.name), task_reductions);
  }
  (void)odsState.addRegion();
}

void TaskGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange task_reduction_vars, /*optional*/::mlir::ArrayAttr task_reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  odsState.addOperands(task_reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(task_reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (task_reductions) {
    odsState.addAttribute(getTaskReductionsAttrName(odsState.name), task_reductions);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskGroupOp::verifyInvariantsImpl() {
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
  ::mlir::Attribute tblgen_task_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getTaskReductionsAttrName()) {
      tblgen_task_reductions = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_task_reductions, "task_reductions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TaskGroupOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TaskGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> task_reduction_varsOperands;
  ::llvm::SMLoc task_reduction_varsOperandsLoc;
  (void)task_reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> task_reduction_varsTypes;
  ::mlir::ArrayAttr task_reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool task_reductionClause = false;
  bool allocateClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("task_reduction"))) {

  if (task_reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`task_reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  task_reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    task_reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, task_reduction_varsOperands, task_reduction_varsTypes, task_reductionsAttr))
      return ::mlir::failure();
    if (task_reductionsAttr)
      result.addAttribute("task_reductions", task_reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(task_reduction_varsOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  if (parser.resolveOperands(task_reduction_varsOperands, task_reduction_varsTypes, task_reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskGroupOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getTaskReductionVars().size() || getTaskReductionsAttr()) {
  _odsPrinter << ' ' << "task_reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getTaskReductionVars(), getTaskReductionVars().getTypes(), getTaskReductionsAttr());
  _odsPrinter << ")";
  }
  if (false || getAllocateVars().size() || getAllocatorsVars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, getAllocateVars(), getAllocateVars().getTypes(), getAllocatorsVars(), getAllocatorsVars().getTypes());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("task_reductions");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskGroupOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskLoopOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TaskLoopOpGenericAdaptorBase::TaskLoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.taskloop", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TaskLoopOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TaskLoopOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr TaskLoopOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TaskLoopOpGenericAdaptorBase::getInclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskLoopOp::getInclusiveAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TaskLoopOpGenericAdaptorBase::getInclusive() {
  auto attr = getInclusiveAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr TaskLoopOpGenericAdaptorBase::getUntiedAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TaskLoopOp::getUntiedAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TaskLoopOpGenericAdaptorBase::getUntied() {
  auto attr = getUntiedAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr TaskLoopOpGenericAdaptorBase::getMergeableAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskLoopOp::getMergeableAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TaskLoopOpGenericAdaptorBase::getMergeable() {
  auto attr = getMergeableAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr TaskLoopOpGenericAdaptorBase::getInReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskLoopOp::getInReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TaskLoopOpGenericAdaptorBase::getInReductions() {
  auto attr = getInReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr TaskLoopOpGenericAdaptorBase::getReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TaskLoopOp::getReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TaskLoopOpGenericAdaptorBase::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::UnitAttr TaskLoopOpGenericAdaptorBase::getNogroupAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskLoopOp::getNogroupAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TaskLoopOpGenericAdaptorBase::getNogroup() {
  auto attr = getNogroupAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &TaskLoopOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange TaskLoopOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
TaskLoopOpAdaptor::TaskLoopOpAdaptor(TaskLoopOp op) : TaskLoopOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TaskLoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_in_reductions;
  ::mlir::Attribute tblgen_inclusive;
  ::mlir::Attribute tblgen_mergeable;
  ::mlir::Attribute tblgen_nogroup;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.taskloop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TaskLoopOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TaskLoopOp::getInReductionsAttrName(*odsOpName)) {
      tblgen_in_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskLoopOp::getInclusiveAttrName(*odsOpName)) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskLoopOp::getMergeableAttrName(*odsOpName)) {
      tblgen_mergeable = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskLoopOp::getNogroupAttrName(*odsOpName)) {
      tblgen_nogroup = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  ::mlir::Attribute tblgen_untied;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TaskLoopOp::getReductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskLoopOp::getUntiedAttrName(*odsOpName)) {
      tblgen_untied = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 12)
      return emitError(loc, "'omp.taskloop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 12 "
                "elements, but got ") << numElements;
  }

  if (tblgen_inclusive && !((tblgen_inclusive.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.taskloop' op ""attribute 'inclusive' failed to satisfy constraint: unit attribute");

  if (tblgen_untied && !((tblgen_untied.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.taskloop' op ""attribute 'untied' failed to satisfy constraint: unit attribute");

  if (tblgen_mergeable && !((tblgen_mergeable.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.taskloop' op ""attribute 'mergeable' failed to satisfy constraint: unit attribute");

  if (tblgen_in_reductions && !(((tblgen_in_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_in_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.taskloop' op ""attribute 'in_reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.taskloop' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_nogroup && !((tblgen_nogroup.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.taskloop' op ""attribute 'nogroup' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range TaskLoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range TaskLoopOp::getLowerBound() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range TaskLoopOp::getUpperBound() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range TaskLoopOp::getStep() {
  return getODSOperands(2);
}

::mlir::TypedValue<::mlir::IntegerType> TaskLoopOp::getIfExpr() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> TaskLoopOp::getFinalExpr() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range TaskLoopOp::getInReductionVars() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range TaskLoopOp::getReductionVars() {
  return getODSOperands(6);
}

::mlir::Value TaskLoopOp::getPriority() {
  auto operands = getODSOperands(7);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range TaskLoopOp::getAllocateVars() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range TaskLoopOp::getAllocatorsVars() {
  return getODSOperands(9);
}

::mlir::Value TaskLoopOp::getGrainSize() {
  auto operands = getODSOperands(10);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Value TaskLoopOp::getNumTasks() {
  auto operands = getODSOperands(11);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::MutableOperandRange TaskLoopOp::getLowerBoundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getUpperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getStepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getFinalExprMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getInReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getPriorityMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getAllocateVarsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getAllocatorsVarsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getGrainSizeMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskLoopOp::getNumTasksMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TaskLoopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskLoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TaskLoopOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr TaskLoopOp::getInclusiveAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInclusiveAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskLoopOp::getInclusive() {
  auto attr = getInclusiveAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr TaskLoopOp::getUntiedAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getUntiedAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskLoopOp::getUntied() {
  auto attr = getUntiedAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr TaskLoopOp::getMergeableAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMergeableAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskLoopOp::getMergeable() {
  auto attr = getMergeableAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr TaskLoopOp::getInReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TaskLoopOp::getInReductions() {
  auto attr = getInReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr TaskLoopOp::getReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TaskLoopOp::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::UnitAttr TaskLoopOp::getNogroupAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNogroupAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskLoopOp::getNogroup() {
  auto attr = getNogroupAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void TaskLoopOp::setInclusiveAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInclusiveAttrName(), attr);
}

void TaskLoopOp::setInclusive(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInclusiveAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInclusiveAttrName());
}

void TaskLoopOp::setUntiedAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUntiedAttrName(), attr);
}

void TaskLoopOp::setUntied(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getUntiedAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getUntiedAttrName());
}

void TaskLoopOp::setMergeableAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getMergeableAttrName(), attr);
}

void TaskLoopOp::setMergeable(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getMergeableAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getMergeableAttrName());
}

void TaskLoopOp::setInReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInReductionsAttrName(), attr);
}

void TaskLoopOp::setReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReductionsAttrName(), attr);
}

void TaskLoopOp::setNogroupAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNogroupAttrName(), attr);
}

void TaskLoopOp::setNogroup(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNogroupAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNogroupAttrName());
}

::mlir::Attribute TaskLoopOp::removeInclusiveAttr() {
  return (*this)->removeAttr(getInclusiveAttrName());
}

::mlir::Attribute TaskLoopOp::removeUntiedAttr() {
  return (*this)->removeAttr(getUntiedAttrName());
}

::mlir::Attribute TaskLoopOp::removeMergeableAttr() {
  return (*this)->removeAttr(getMergeableAttrName());
}

::mlir::Attribute TaskLoopOp::removeInReductionsAttr() {
  return (*this)->removeAttr(getInReductionsAttrName());
}

::mlir::Attribute TaskLoopOp::removeReductionsAttr() {
  return (*this)->removeAttr(getReductionsAttrName());
}

::mlir::Attribute TaskLoopOp::removeNogroupAttr() {
  return (*this)->removeAttr(getNogroupAttrName());
}

void TaskLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::UnitAttr inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/::mlir::UnitAttr nogroup) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  odsState.addOperands(reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  if (grain_size)
    odsState.addOperands(grain_size);
  if (num_tasks)
    odsState.addOperands(num_tasks);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), (if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), static_cast<int32_t>(reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), (grain_size ? 1 : 0), (num_tasks ? 1 : 0)}));
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  }
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), untied);
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), mergeable);
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nogroup) {
    odsState.addAttribute(getNogroupAttrName(odsState.name), nogroup);
  }
  (void)odsState.addRegion();
}

void TaskLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::UnitAttr inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/::mlir::UnitAttr nogroup) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  odsState.addOperands(reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  if (grain_size)
    odsState.addOperands(grain_size);
  if (num_tasks)
    odsState.addOperands(num_tasks);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), (if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), static_cast<int32_t>(reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), (grain_size ? 1 : 0), (num_tasks ? 1 : 0)}));
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  }
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), untied);
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), mergeable);
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nogroup) {
    odsState.addAttribute(getNogroupAttrName(odsState.name), nogroup);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/bool inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/bool nogroup) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  odsState.addOperands(reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  if (grain_size)
    odsState.addOperands(grain_size);
  if (num_tasks)
    odsState.addOperands(num_tasks);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), (if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), static_cast<int32_t>(reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), (grain_size ? 1 : 0), (num_tasks ? 1 : 0)}));
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), ((inclusive) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), ((untied) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), ((mergeable) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nogroup) {
    odsState.addAttribute(getNogroupAttrName(odsState.name), ((nogroup) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
}

void TaskLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/bool inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/bool nogroup) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  odsState.addOperands(reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  if (grain_size)
    odsState.addOperands(grain_size);
  if (num_tasks)
    odsState.addOperands(num_tasks);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), (if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), static_cast<int32_t>(reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), (grain_size ? 1 : 0), (num_tasks ? 1 : 0)}));
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), ((inclusive) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), ((untied) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), ((mergeable) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (nogroup) {
    odsState.addAttribute(getNogroupAttrName(odsState.name), ((nogroup) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskLoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskLoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_in_reductions;
  ::mlir::Attribute tblgen_inclusive;
  ::mlir::Attribute tblgen_mergeable;
  ::mlir::Attribute tblgen_nogroup;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getInReductionsAttrName()) {
      tblgen_in_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getInclusiveAttrName()) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMergeableAttrName()) {
      tblgen_mergeable = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNogroupAttrName()) {
      tblgen_nogroup = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  ::mlir::Attribute tblgen_untied;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUntiedAttrName()) {
      tblgen_untied = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 12)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 12 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_inclusive, "inclusive")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_untied, "untied")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_mergeable, "mergeable")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_in_reductions, "in_reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nogroup, "nogroup")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    if (valueGroup7.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup7.size();
    }

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    if (valueGroup10.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup10.size();
    }

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    if (valueGroup11.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup11.size();
    }

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((this->getODSOperands(0).getType()) == (this->getODSOperands(1).getType()) && (this->getODSOperands(1).getType()) == (this->getODSOperands(2).getType()) && (this->getODSOperands(2).getType()) == (this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that all of {lowerBound, upperBound, step} have same type");
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TaskLoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TaskLoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> final_exprOperands;
  ::llvm::SMLoc final_exprOperandsLoc;
  (void)final_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> in_reduction_varsOperands;
  ::llvm::SMLoc in_reduction_varsOperandsLoc;
  (void)in_reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> in_reduction_varsTypes;
  ::mlir::ArrayAttr in_reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> priorityOperands;
  ::llvm::SMLoc priorityOperandsLoc;
  (void)priorityOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> priorityTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> grain_sizeOperands;
  ::llvm::SMLoc grain_sizeOperandsLoc;
  (void)grain_sizeOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> grain_sizeTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> num_tasksOperands;
  ::llvm::SMLoc num_tasksOperandsLoc;
  (void)num_tasksOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> num_tasksTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lowerBoundOperands;
  ::llvm::SMLoc lowerBoundOperandsLoc;
  (void)lowerBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> upperBoundOperands;
  ::llvm::SMLoc upperBoundOperandsLoc;
  (void)upperBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stepOperands;
  ::llvm::SMLoc stepOperandsLoc;
  (void)stepOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> stepTypes;
  ::mlir::UnitAttr inclusiveAttr;
  bool ifClause = false;
  bool finalClause = false;
  bool untiedClause = false;
  bool mergeableClause = false;
  bool in_reductionClause = false;
  bool reductionClause = false;
  bool priorityClause = false;
  bool allocateClause = false;
  bool grain_sizeClause = false;
  bool num_tasksClause = false;
  bool nogroupClause = false;
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("final"))) {

  if (finalClause) {
    return parser.emitError(parser.getNameLoc())
          << "`final` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  finalClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    final_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      final_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("untied"))) {

  if (untiedClause) {
    return parser.emitError(parser.getNameLoc())
          << "`untied` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  untiedClause = true;
  result.addAttribute("untied", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("mergeable"))) {

  if (mergeableClause) {
    return parser.emitError(parser.getNameLoc())
          << "`mergeable` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  mergeableClause = true;
  result.addAttribute("mergeable", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("in_reduction"))) {

  if (in_reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`in_reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  in_reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    in_reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, in_reduction_varsOperands, in_reduction_varsTypes, in_reductionsAttr))
      return ::mlir::failure();
    if (in_reductionsAttr)
      result.addAttribute("in_reductions", in_reductionsAttr);
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
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("priority"))) {

  if (priorityClause) {
    return parser.emitError(parser.getNameLoc())
          << "`priority` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  priorityClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    priorityOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      priorityOperands.push_back(operand);
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
      priorityTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("grain_size"))) {

  if (grain_sizeClause) {
    return parser.emitError(parser.getNameLoc())
          << "`grain_size` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  grain_sizeClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    grain_sizeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      grain_sizeOperands.push_back(operand);
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
      grain_sizeTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_tasks"))) {

  if (num_tasksClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_tasks` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_tasksClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    num_tasksOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      num_tasksOperands.push_back(operand);
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
      num_tasksTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nogroup"))) {

  if (nogroupClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nogroup` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nogroupClause = true;
  result.addAttribute("nogroup", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}
  if (parser.parseKeyword("for"))
    return ::mlir::failure();
  {
    lowerBoundOperandsLoc = parser.getCurrentLocation();
    upperBoundOperandsLoc = parser.getCurrentLocation();
    stepOperandsLoc = parser.getCurrentLocation();
    if (parseLoopControl(parser, *regionRegion, lowerBoundOperands, upperBoundOperands, stepOperands, stepTypes, inclusiveAttr))
      return ::mlir::failure();
    if (inclusiveAttr)
      result.addAttribute("inclusive", inclusiveAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(lowerBoundOperands.size()), static_cast<int32_t>(upperBoundOperands.size()), static_cast<int32_t>(stepOperands.size()), static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(final_exprOperands.size()), static_cast<int32_t>(in_reduction_varsOperands.size()), static_cast<int32_t>(reduction_varsOperands.size()), static_cast<int32_t>(priorityOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size()), static_cast<int32_t>(grain_sizeOperands.size()), static_cast<int32_t>(num_tasksOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(lowerBoundOperands, stepTypes, lowerBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperBoundOperands, stepTypes, upperBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stepOperands, stepTypes, stepOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(final_exprOperands, odsBuildableType0, final_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(in_reduction_varsOperands, in_reduction_varsTypes, in_reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(priorityOperands, priorityTypes, priorityOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(grain_sizeOperands, grain_sizeTypes, grain_sizeOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(num_tasksOperands, num_tasksTypes, num_tasksOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskLoopOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getFinalExpr()) {
  _odsPrinter << ' ' << "final";
  _odsPrinter << "(";
  if (::mlir::Value value = getFinalExpr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getUntiedAttr()) {
  _odsPrinter << ' ' << "untied";
  }
  if (false || getMergeableAttr()) {
  _odsPrinter << ' ' << "mergeable";
  }
  if (false || getInReductionVars().size() || getInReductionsAttr()) {
  _odsPrinter << ' ' << "in_reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getInReductionVars(), getInReductionVars().getTypes(), getInReductionsAttr());
  _odsPrinter << ")";
  }
  if (false || getReductionVars().size() || getReductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getReductionVars(), getReductionVars().getTypes(), getReductionsAttr());
  _odsPrinter << ")";
  }
  if (false || getPriority()) {
  _odsPrinter << ' ' << "priority";
  _odsPrinter << "(";
  if (::mlir::Value value = getPriority())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getPriority() ? ::llvm::ArrayRef<::mlir::Type>(getPriority().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getAllocateVars().size() || getAllocatorsVars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, getAllocateVars(), getAllocateVars().getTypes(), getAllocatorsVars(), getAllocatorsVars().getTypes());
  _odsPrinter << ")";
  }
  if (false || getGrainSize()) {
  _odsPrinter << ' ' << "grain_size";
  _odsPrinter << "(";
  if (::mlir::Value value = getGrainSize())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getGrainSize() ? ::llvm::ArrayRef<::mlir::Type>(getGrainSize().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNumTasks()) {
  _odsPrinter << ' ' << "num_tasks";
  _odsPrinter << "(";
  if (::mlir::Value value = getNumTasks())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getNumTasks() ? ::llvm::ArrayRef<::mlir::Type>(getNumTasks().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || getNogroupAttr()) {
  _odsPrinter << ' ' << "nogroup";
  }
  _odsPrinter << ' ' << "for";
  _odsPrinter << ' ';
  printLoopControl(_odsPrinter, *this, getRegion(), getLowerBound(), getUpperBound(), getStep(), getStep().getTypes(), getInclusiveAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("untied");
  elidedAttrs.push_back("mergeable");
  elidedAttrs.push_back("in_reductions");
  elidedAttrs.push_back("reductions");
  elidedAttrs.push_back("nogroup");
  elidedAttrs.push_back("inclusive");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInclusiveAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inclusive");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getUntiedAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("untied");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getMergeableAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("mergeable");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNogroupAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nogroup");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TaskOpGenericAdaptorBase::TaskOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.task", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TaskOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TaskOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr TaskOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TaskOpGenericAdaptorBase::getUntiedAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TaskOp::getUntiedAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TaskOpGenericAdaptorBase::getUntied() {
  auto attr = getUntiedAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr TaskOpGenericAdaptorBase::getMergeableAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskOp::getMergeableAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TaskOpGenericAdaptorBase::getMergeable() {
  auto attr = getMergeableAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr TaskOpGenericAdaptorBase::getInReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskOp::getInReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TaskOpGenericAdaptorBase::getInReductions() {
  auto attr = getInReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr TaskOpGenericAdaptorBase::getDependsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskOp::getDependsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TaskOpGenericAdaptorBase::getDepends() {
  auto attr = getDependsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::Region &TaskOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange TaskOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
TaskOpAdaptor::TaskOpAdaptor(TaskOp op) : TaskOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TaskOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_depends;
  ::mlir::Attribute tblgen_in_reductions;
  ::mlir::Attribute tblgen_mergeable;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.task' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TaskOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TaskOp::getDependsAttrName(*odsOpName)) {
      tblgen_depends = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskOp::getInReductionsAttrName(*odsOpName)) {
      tblgen_in_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskOp::getMergeableAttrName(*odsOpName)) {
      tblgen_mergeable = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_untied;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TaskOp::getUntiedAttrName(*odsOpName)) {
      tblgen_untied = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'omp.task' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_untied && !((tblgen_untied.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.task' op ""attribute 'untied' failed to satisfy constraint: unit attribute");

  if (tblgen_mergeable && !((tblgen_mergeable.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.task' op ""attribute 'mergeable' failed to satisfy constraint: unit attribute");

  if (tblgen_in_reductions && !(((tblgen_in_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_in_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.task' op ""attribute 'in_reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_depends && !(((tblgen_depends.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_depends.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::omp::ClauseTaskDependAttr>())); }))))
    return emitError(loc, "'omp.task' op ""attribute 'depends' failed to satisfy constraint: clause_task_depend array attr");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range TaskOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> TaskOp::getIfExpr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::TypedValue<::mlir::IntegerType> TaskOp::getFinalExpr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range TaskOp::getInReductionVars() {
  return getODSOperands(2);
}

::mlir::TypedValue<::mlir::IntegerType> TaskOp::getPriority() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
}

::mlir::Operation::operand_range TaskOp::getDependVars() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range TaskOp::getAllocateVars() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range TaskOp::getAllocatorsVars() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange TaskOp::getIfExprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::getFinalExprMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::getInReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::getPriorityMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::getDependVarsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::getAllocateVarsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::getAllocatorsVarsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TaskOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TaskOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr TaskOp::getUntiedAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getUntiedAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskOp::getUntied() {
  auto attr = getUntiedAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr TaskOp::getMergeableAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMergeableAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskOp::getMergeable() {
  auto attr = getMergeableAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr TaskOp::getInReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TaskOp::getInReductions() {
  auto attr = getInReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr TaskOp::getDependsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDependsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TaskOp::getDepends() {
  auto attr = getDependsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void TaskOp::setUntiedAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUntiedAttrName(), attr);
}

void TaskOp::setUntied(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getUntiedAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getUntiedAttrName());
}

void TaskOp::setMergeableAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getMergeableAttrName(), attr);
}

void TaskOp::setMergeable(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getMergeableAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getMergeableAttrName());
}

void TaskOp::setInReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInReductionsAttrName(), attr);
}

void TaskOp::setDependsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDependsAttrName(), attr);
}

::mlir::Attribute TaskOp::removeUntiedAttr() {
  return (*this)->removeAttr(getUntiedAttrName());
}

::mlir::Attribute TaskOp::removeMergeableAttr() {
  return (*this)->removeAttr(getMergeableAttrName());
}

::mlir::Attribute TaskOp::removeInReductionsAttr() {
  return (*this)->removeAttr(getInReductionsAttrName());
}

::mlir::Attribute TaskOp::removeDependsAttr() {
  return (*this)->removeAttr(getDependsAttrName());
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(depend_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(depend_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), untied);
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), mergeable);
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (depends) {
    odsState.addAttribute(getDependsAttrName(odsState.name), depends);
  }
  (void)odsState.addRegion();
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(depend_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(depend_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), untied);
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), mergeable);
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (depends) {
    odsState.addAttribute(getDependsAttrName(odsState.name), depends);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(depend_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(depend_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), ((untied) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), ((mergeable) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (depends) {
    odsState.addAttribute(getDependsAttrName(odsState.name), depends);
  }
  (void)odsState.addRegion();
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, /*optional*/::mlir::ArrayAttr depends, ::mlir::ValueRange depend_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(depend_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(depend_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
    odsState.addAttribute(getUntiedAttrName(odsState.name), ((untied) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (mergeable) {
    odsState.addAttribute(getMergeableAttrName(odsState.name), ((mergeable) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (in_reductions) {
    odsState.addAttribute(getInReductionsAttrName(odsState.name), in_reductions);
  }
  if (depends) {
    odsState.addAttribute(getDependsAttrName(odsState.name), depends);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_depends;
  ::mlir::Attribute tblgen_in_reductions;
  ::mlir::Attribute tblgen_mergeable;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDependsAttrName()) {
      tblgen_depends = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getInReductionsAttrName()) {
      tblgen_in_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMergeableAttrName()) {
      tblgen_mergeable = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_untied;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getUntiedAttrName()) {
      tblgen_untied = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_untied, "untied")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_mergeable, "mergeable")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_in_reductions, "in_reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps14(*this, tblgen_depends, "depends")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TaskOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TaskOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> final_exprOperands;
  ::llvm::SMLoc final_exprOperandsLoc;
  (void)final_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> in_reduction_varsOperands;
  ::llvm::SMLoc in_reduction_varsOperandsLoc;
  (void)in_reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> in_reduction_varsTypes;
  ::mlir::ArrayAttr in_reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> priorityOperands;
  ::llvm::SMLoc priorityOperandsLoc;
  (void)priorityOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> depend_varsOperands;
  ::llvm::SMLoc depend_varsOperandsLoc;
  (void)depend_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> depend_varsTypes;
  ::mlir::ArrayAttr dependsAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool finalClause = false;
  bool untiedClause = false;
  bool mergeableClause = false;
  bool in_reductionClause = false;
  bool priorityClause = false;
  bool allocateClause = false;
  bool dependClause = false;
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
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("final"))) {

  if (finalClause) {
    return parser.emitError(parser.getNameLoc())
          << "`final` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  finalClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    final_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      final_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("untied"))) {

  if (untiedClause) {
    return parser.emitError(parser.getNameLoc())
          << "`untied` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  untiedClause = true;
  result.addAttribute("untied", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("mergeable"))) {

  if (mergeableClause) {
    return parser.emitError(parser.getNameLoc())
          << "`mergeable` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  mergeableClause = true;
  result.addAttribute("mergeable", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("in_reduction"))) {

  if (in_reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`in_reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  in_reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    in_reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, in_reduction_varsOperands, in_reduction_varsTypes, in_reductionsAttr))
      return ::mlir::failure();
    if (in_reductionsAttr)
      result.addAttribute("in_reductions", in_reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("priority"))) {

  if (priorityClause) {
    return parser.emitError(parser.getNameLoc())
          << "`priority` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  priorityClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    priorityOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      priorityOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("depend"))) {

  if (dependClause) {
    return parser.emitError(parser.getNameLoc())
          << "`depend` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  dependClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    depend_varsOperandsLoc = parser.getCurrentLocation();
    if (parseDependVarList(parser, depend_varsOperands, depend_varsTypes, dependsAttr))
      return ::mlir::failure();
    if (dependsAttr)
      result.addAttribute("depends", dependsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(final_exprOperands.size()), static_cast<int32_t>(in_reduction_varsOperands.size()), static_cast<int32_t>(priorityOperands.size()), static_cast<int32_t>(depend_varsOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, if_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(final_exprOperands, odsBuildableType0, final_exprOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(in_reduction_varsOperands, in_reduction_varsTypes, in_reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(priorityOperands, odsBuildableType1, priorityOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(depend_varsOperands, depend_varsTypes, depend_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getIfExpr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = getIfExpr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getFinalExpr()) {
  _odsPrinter << ' ' << "final";
  _odsPrinter << "(";
  if (::mlir::Value value = getFinalExpr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getUntiedAttr()) {
  _odsPrinter << ' ' << "untied";
  }
  if (false || getMergeableAttr()) {
  _odsPrinter << ' ' << "mergeable";
  }
  if (false || getInReductionVars().size() || getInReductionsAttr()) {
  _odsPrinter << ' ' << "in_reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getInReductionVars(), getInReductionVars().getTypes(), getInReductionsAttr());
  _odsPrinter << ")";
  }
  if (false || getPriority()) {
  _odsPrinter << ' ' << "priority";
  _odsPrinter << "(";
  if (::mlir::Value value = getPriority())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || getAllocateVars().size() || getAllocatorsVars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, getAllocateVars(), getAllocateVars().getTypes(), getAllocatorsVars(), getAllocatorsVars().getTypes());
  _odsPrinter << ")";
  }
  if (false || getDependVars().size() || getDependsAttr()) {
  _odsPrinter << ' ' << "depend";
  _odsPrinter << "(";
  printDependVarList(_odsPrinter, *this, getDependVars(), getDependVars().getTypes(), getDependsAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("untied");
  elidedAttrs.push_back("mergeable");
  elidedAttrs.push_back("in_reductions");
  elidedAttrs.push_back("depends");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getUntiedAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("untied");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getMergeableAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("mergeable");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TaskwaitOpGenericAdaptorBase::TaskwaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.taskwait", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TaskwaitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TaskwaitOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TaskwaitOpAdaptor::TaskwaitOpAdaptor(TaskwaitOp op) : TaskwaitOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TaskwaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskwaitOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TaskwaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TaskwaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskwaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TaskwaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TaskwaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskwaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskwaitOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TaskwaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TaskwaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskwaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TaskyieldOpGenericAdaptorBase::TaskyieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.taskyield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TaskyieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TaskyieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TaskyieldOpAdaptor::TaskyieldOpAdaptor(TaskyieldOp op) : TaskyieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TaskyieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskyieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TaskyieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TaskyieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskyieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TaskyieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TaskyieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskyieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskyieldOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TaskyieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TaskyieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskyieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TerminatorOpGenericAdaptorBase::TerminatorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.terminator", odsAttrs.getContext());
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

void TerminatorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ThreadprivateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ThreadprivateOpGenericAdaptorBase::ThreadprivateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.threadprivate", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ThreadprivateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ThreadprivateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ThreadprivateOpAdaptor::ThreadprivateOpAdaptor(ThreadprivateOp op) : ThreadprivateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ThreadprivateOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ThreadprivateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadprivateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> ThreadprivateOp::getSymAddr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ThreadprivateOp::getSymAddrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ThreadprivateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadprivateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::omp::PointerLikeType> ThreadprivateOp::getTlsAddr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::omp::PointerLikeType>>(*getODSResults(0).begin());
}

void ThreadprivateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tls_addr, ::mlir::Value sym_addr) {
  odsState.addOperands(sym_addr);
  odsState.addTypes(tls_addr);
}

void ThreadprivateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sym_addr) {
  odsState.addOperands(sym_addr);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadprivateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadprivateOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {sym_addr, tls_addr} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult ThreadprivateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ThreadprivateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sym_addrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sym_addrOperands(sym_addrRawOperands);  ::llvm::SMLoc sym_addrOperandsLoc;
  (void)sym_addrOperandsLoc;
  ::mlir::Type sym_addrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sym_addrTypes(sym_addrRawTypes);
  ::mlir::Type tls_addrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tls_addrTypes(tls_addrRawTypes);

  sym_addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sym_addrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sym_addrRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tls_addrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(tls_addrTypes);
  if (parser.resolveOperands(sym_addrOperands, sym_addrTypes, sym_addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ThreadprivateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSymAddr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSymAddr().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getTlsAddr().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ThreadprivateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WsLoopOpGenericAdaptorBase::WsLoopOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.wsloop", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> WsLoopOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, WsLoopOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr WsLoopOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr WsLoopOpGenericAdaptorBase::getReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::getReductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > WsLoopOpGenericAdaptorBase::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::omp::ClauseScheduleKindAttr WsLoopOpGenericAdaptorBase::getScheduleValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::getScheduleValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseScheduleKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseScheduleKind> WsLoopOpGenericAdaptorBase::getScheduleVal() {
  auto attr = getScheduleValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseScheduleKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::omp::ScheduleModifierAttr WsLoopOpGenericAdaptorBase::getScheduleModifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::getScheduleModifierAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ScheduleModifierAttr>();
  return attr;
}

::std::optional<::mlir::omp::ScheduleModifier> WsLoopOpGenericAdaptorBase::getScheduleModifier() {
  auto attr = getScheduleModifierAttr();
  return attr ? ::std::optional<::mlir::omp::ScheduleModifier>(attr.getValue()) : (::std::nullopt);
}

::mlir::UnitAttr WsLoopOpGenericAdaptorBase::getSimdModifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::getSimdModifierAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool WsLoopOpGenericAdaptorBase::getSimdModifier() {
  auto attr = getSimdModifierAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr WsLoopOpGenericAdaptorBase::getNowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WsLoopOp::getNowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool WsLoopOpGenericAdaptorBase::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOpGenericAdaptorBase::getOrderedValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::getOrderedValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> WsLoopOpGenericAdaptorBase::getOrderedVal() {
  auto attr = getOrderedValAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::omp::ClauseOrderKindAttr WsLoopOpGenericAdaptorBase::getOrderValAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::getOrderValAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
  return attr;
}

::std::optional<::mlir::omp::ClauseOrderKind> WsLoopOpGenericAdaptorBase::getOrderVal() {
  auto attr = getOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::UnitAttr WsLoopOpGenericAdaptorBase::getInclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WsLoopOp::getInclusiveAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool WsLoopOpGenericAdaptorBase::getInclusive() {
  auto attr = getInclusiveAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &WsLoopOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange WsLoopOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
WsLoopOpAdaptor::WsLoopOpAdaptor(WsLoopOp op) : WsLoopOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult WsLoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_inclusive;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.wsloop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == WsLoopOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WsLoopOp::getInclusiveAttrName(*odsOpName)) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::getNowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_order_val;
  ::mlir::Attribute tblgen_ordered_val;
  ::mlir::Attribute tblgen_reductions;
  ::mlir::Attribute tblgen_schedule_modifier;
  ::mlir::Attribute tblgen_schedule_val;
  ::mlir::Attribute tblgen_simd_modifier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == WsLoopOp::getOrderValAttrName(*odsOpName)) {
      tblgen_order_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::getOrderedValAttrName(*odsOpName)) {
      tblgen_ordered_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::getReductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::getScheduleModifierAttrName(*odsOpName)) {
      tblgen_schedule_modifier = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::getScheduleValAttrName(*odsOpName)) {
      tblgen_schedule_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::getSimdModifierAttrName(*odsOpName)) {
      tblgen_simd_modifier = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'omp.wsloop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.wsloop' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_schedule_val && !((tblgen_schedule_val.isa<::mlir::omp::ClauseScheduleKindAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_val' failed to satisfy constraint: ScheduleKind Clause");

  if (tblgen_schedule_modifier && !((tblgen_schedule_modifier.isa<::mlir::omp::ScheduleModifierAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_modifier' failed to satisfy constraint: OpenMP Schedule Modifier");

  if (tblgen_simd_modifier && !((tblgen_simd_modifier.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'simd_modifier' failed to satisfy constraint: unit attribute");

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");

  if (tblgen_ordered_val && !((((tblgen_ordered_val.isa<::mlir::IntegerAttr>())) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'omp.wsloop' op ""attribute 'ordered_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");

  if (tblgen_order_val && !((tblgen_order_val.isa<::mlir::omp::ClauseOrderKindAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'order_val' failed to satisfy constraint: OrderKind Clause");

  if (tblgen_inclusive && !((tblgen_inclusive.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'inclusive' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WsLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range WsLoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WsLoopOp::getLowerBound() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range WsLoopOp::getUpperBound() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range WsLoopOp::getStep() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range WsLoopOp::getLinearVars() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range WsLoopOp::getLinearStepVars() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range WsLoopOp::getReductionVars() {
  return getODSOperands(5);
}

::mlir::Value WsLoopOp::getScheduleChunkVar() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::MutableOperandRange WsLoopOp::getLowerBoundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::getUpperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::getStepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::getLinearVarsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::getLinearStepVarsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::getReductionVarsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::getScheduleChunkVarMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> WsLoopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WsLoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &WsLoopOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr WsLoopOp::getReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getReductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > WsLoopOp::getReductions() {
  auto attr = getReductionsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::omp::ClauseScheduleKindAttr WsLoopOp::getScheduleValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getScheduleValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseScheduleKindAttr>();
}

::std::optional<::mlir::omp::ClauseScheduleKind> WsLoopOp::getScheduleVal() {
  auto attr = getScheduleValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseScheduleKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::omp::ScheduleModifierAttr WsLoopOp::getScheduleModifierAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getScheduleModifierAttrName()).dyn_cast_or_null<::mlir::omp::ScheduleModifierAttr>();
}

::std::optional<::mlir::omp::ScheduleModifier> WsLoopOp::getScheduleModifier() {
  auto attr = getScheduleModifierAttr();
  return attr ? ::std::optional<::mlir::omp::ScheduleModifier>(attr.getValue()) : (::std::nullopt);
}

::mlir::UnitAttr WsLoopOp::getSimdModifierAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSimdModifierAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::getSimdModifier() {
  auto attr = getSimdModifierAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr WsLoopOp::getNowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::getNowait() {
  auto attr = getNowaitAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOp::getOrderedValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOrderedValAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> WsLoopOp::getOrderedVal() {
  auto attr = getOrderedValAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::omp::ClauseOrderKindAttr WsLoopOp::getOrderValAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOrderValAttrName()).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
}

::std::optional<::mlir::omp::ClauseOrderKind> WsLoopOp::getOrderVal() {
  auto attr = getOrderValAttr();
  return attr ? ::std::optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::std::nullopt);
}

::mlir::UnitAttr WsLoopOp::getInclusiveAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInclusiveAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::getInclusive() {
  auto attr = getInclusiveAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void WsLoopOp::setReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReductionsAttrName(), attr);
}

void WsLoopOp::setScheduleValAttr(::mlir::omp::ClauseScheduleKindAttr attr) {
  (*this)->setAttr(getScheduleValAttrName(), attr);
}

void WsLoopOp::setScheduleVal(::std::optional<::mlir::omp::ClauseScheduleKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getScheduleValAttrName(), ::mlir::omp::ClauseScheduleKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getScheduleValAttrName());
}

void WsLoopOp::setScheduleModifierAttr(::mlir::omp::ScheduleModifierAttr attr) {
  (*this)->setAttr(getScheduleModifierAttrName(), attr);
}

void WsLoopOp::setScheduleModifier(::std::optional<::mlir::omp::ScheduleModifier> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getScheduleModifierAttrName(), ::mlir::omp::ScheduleModifierAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getScheduleModifierAttrName());
}

void WsLoopOp::setSimdModifierAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getSimdModifierAttrName(), attr);
}

void WsLoopOp::setSimdModifier(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSimdModifierAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getSimdModifierAttrName());
}

void WsLoopOp::setNowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNowaitAttrName(), attr);
}

void WsLoopOp::setNowait(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNowaitAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNowaitAttrName());
}

void WsLoopOp::setOrderedValAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getOrderedValAttrName(), attr);
}

void WsLoopOp::setOrderedVal(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOrderedValAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getOrderedValAttrName());
}

void WsLoopOp::setOrderValAttr(::mlir::omp::ClauseOrderKindAttr attr) {
  (*this)->setAttr(getOrderValAttrName(), attr);
}

void WsLoopOp::setOrderVal(::std::optional<::mlir::omp::ClauseOrderKind> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOrderValAttrName(), ::mlir::omp::ClauseOrderKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getOrderValAttrName());
}

void WsLoopOp::setInclusiveAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInclusiveAttrName(), attr);
}

void WsLoopOp::setInclusive(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInclusiveAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInclusiveAttrName());
}

::mlir::Attribute WsLoopOp::removeReductionsAttr() {
  return (*this)->removeAttr(getReductionsAttrName());
}

::mlir::Attribute WsLoopOp::removeScheduleValAttr() {
  return (*this)->removeAttr(getScheduleValAttrName());
}

::mlir::Attribute WsLoopOp::removeScheduleModifierAttr() {
  return (*this)->removeAttr(getScheduleModifierAttrName());
}

::mlir::Attribute WsLoopOp::removeSimdModifierAttr() {
  return (*this)->removeAttr(getSimdModifierAttrName());
}

::mlir::Attribute WsLoopOp::removeNowaitAttr() {
  return (*this)->removeAttr(getNowaitAttrName());
}

::mlir::Attribute WsLoopOp::removeOrderedValAttr() {
  return (*this)->removeAttr(getOrderedValAttrName());
}

::mlir::Attribute WsLoopOp::removeOrderValAttr() {
  return (*this)->removeAttr(getOrderValAttrName());
}

::mlir::Attribute WsLoopOp::removeInclusiveAttr() {
  return (*this)->removeAttr(getInclusiveAttrName());
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
    odsState.addAttribute(getScheduleValAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
    odsState.addAttribute(getScheduleModifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
    odsState.addAttribute(getSimdModifierAttrName(odsState.name), simd_modifier);
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  if (ordered_val) {
    odsState.addAttribute(getOrderedValAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  }
  (void)odsState.addRegion();
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
    odsState.addAttribute(getScheduleValAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
    odsState.addAttribute(getScheduleModifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
    odsState.addAttribute(getSimdModifierAttrName(odsState.name), simd_modifier);
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), nowait);
  }
  if (ordered_val) {
    odsState.addAttribute(getOrderedValAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
    odsState.addAttribute(getScheduleValAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
    odsState.addAttribute(getScheduleModifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
    odsState.addAttribute(getSimdModifierAttrName(odsState.name), ((simd_modifier) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (ordered_val) {
    odsState.addAttribute(getOrderedValAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), ((inclusive) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
    odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
    odsState.addAttribute(getScheduleValAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
    odsState.addAttribute(getScheduleModifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
    odsState.addAttribute(getSimdModifierAttrName(odsState.name), ((simd_modifier) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (nowait) {
    odsState.addAttribute(getNowaitAttrName(odsState.name), ((nowait) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (ordered_val) {
    odsState.addAttribute(getOrderedValAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
    odsState.addAttribute(getOrderValAttrName(odsState.name), order_val);
  }
  if (inclusive) {
    odsState.addAttribute(getInclusiveAttrName(odsState.name), ((inclusive) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WsLoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WsLoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_inclusive;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getInclusiveAttrName()) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_order_val;
  ::mlir::Attribute tblgen_ordered_val;
  ::mlir::Attribute tblgen_reductions;
  ::mlir::Attribute tblgen_schedule_modifier;
  ::mlir::Attribute tblgen_schedule_val;
  ::mlir::Attribute tblgen_simd_modifier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getOrderValAttrName()) {
      tblgen_order_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOrderedValAttrName()) {
      tblgen_ordered_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getReductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getScheduleModifierAttrName()) {
      tblgen_schedule_modifier = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getScheduleValAttrName()) {
      tblgen_schedule_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSimdModifierAttrName()) {
      tblgen_simd_modifier = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps15(*this, tblgen_schedule_val, "schedule_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps16(*this, tblgen_schedule_modifier, "schedule_modifier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_simd_modifier, "simd_modifier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_ordered_val, "ordered_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps12(*this, tblgen_order_val, "order_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_inclusive, "inclusive")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    if (valueGroup6.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup6.size();
    }

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((this->getODSOperands(0).getType()) == (this->getODSOperands(1).getType()) && (this->getODSOperands(1).getType()) == (this->getODSOperands(2).getType()) && (this->getODSOperands(2).getType()) == (this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that all of {lowerBound, upperBound, step} have same type");
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult WsLoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WsLoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> linear_varsOperands;
  ::llvm::SMLoc linear_varsOperandsLoc;
  (void)linear_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> linear_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> linear_step_varsOperands;
  ::llvm::SMLoc linear_step_varsOperandsLoc;
  (void)linear_step_varsOperandsLoc;
  ::mlir::omp::ClauseScheduleKindAttr schedule_valAttr;
  ::mlir::omp::ScheduleModifierAttr schedule_modifierAttr;
  ::mlir::UnitAttr simd_modifierAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> schedule_chunk_varOperands;
  ::llvm::SMLoc schedule_chunk_varOperandsLoc;
  (void)schedule_chunk_varOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> schedule_chunk_varTypes;
  ::mlir::IntegerAttr ordered_valAttr;
  ::mlir::omp::ClauseOrderKindAttr order_valAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lowerBoundOperands;
  ::llvm::SMLoc lowerBoundOperandsLoc;
  (void)lowerBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> upperBoundOperands;
  ::llvm::SMLoc upperBoundOperandsLoc;
  (void)upperBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stepOperands;
  ::llvm::SMLoc stepOperandsLoc;
  (void)stepOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> stepTypes;
  ::mlir::UnitAttr inclusiveAttr;
  bool linearClause = false;
  bool scheduleClause = false;
  bool nowaitClause = false;
  bool orderedClause = false;
  bool orderClause = false;
  bool reductionClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("linear"))) {

  if (linearClause) {
    return parser.emitError(parser.getNameLoc())
          << "`linear` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  linearClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    linear_varsOperandsLoc = parser.getCurrentLocation();
    linear_step_varsOperandsLoc = parser.getCurrentLocation();
    if (parseLinearClause(parser, linear_varsOperands, linear_varsTypes, linear_step_varsOperands))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("schedule"))) {

  if (scheduleClause) {
    return parser.emitError(parser.getNameLoc())
          << "`schedule` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  scheduleClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    schedule_chunk_varOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> schedule_chunk_varOperand;
    ::mlir::Type schedule_chunk_varType;
    if (parseScheduleClause(parser, schedule_valAttr, schedule_modifierAttr, simd_modifierAttr, schedule_chunk_varOperand, schedule_chunk_varType))
      return ::mlir::failure();
    if (schedule_valAttr)
      result.addAttribute("schedule_val", schedule_valAttr);
    if (schedule_modifierAttr)
      result.addAttribute("schedule_modifier", schedule_modifierAttr);
    if (simd_modifierAttr)
      result.addAttribute("simd_modifier", simd_modifierAttr);
    if (schedule_chunk_varOperand.has_value())
      schedule_chunk_varOperands.push_back(*schedule_chunk_varOperand);
    if (schedule_chunk_varType)
      schedule_chunk_varTypes.push_back(schedule_chunk_varType);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("ordered"))) {

  if (orderedClause) {
    return parser.emitError(parser.getNameLoc())
          << "`ordered` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  orderedClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(ordered_valAttr, parser.getBuilder().getIntegerType(64), "ordered_val", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("order"))) {

  if (orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, order_valAttr))
      return ::mlir::failure();
    if (order_valAttr)
      result.addAttribute("order_val", order_valAttr);
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
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseKeyword("for"))
    return ::mlir::failure();
  {
    lowerBoundOperandsLoc = parser.getCurrentLocation();
    upperBoundOperandsLoc = parser.getCurrentLocation();
    stepOperandsLoc = parser.getCurrentLocation();
    if (parseLoopControl(parser, *regionRegion, lowerBoundOperands, upperBoundOperands, stepOperands, stepTypes, inclusiveAttr))
      return ::mlir::failure();
    if (inclusiveAttr)
      result.addAttribute("inclusive", inclusiveAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(lowerBoundOperands.size()), static_cast<int32_t>(upperBoundOperands.size()), static_cast<int32_t>(stepOperands.size()), static_cast<int32_t>(linear_varsOperands.size()), static_cast<int32_t>(linear_step_varsOperands.size()), static_cast<int32_t>(reduction_varsOperands.size()), static_cast<int32_t>(schedule_chunk_varOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(lowerBoundOperands, stepTypes, lowerBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperBoundOperands, stepTypes, upperBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stepOperands, stepTypes, stepOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(linear_varsOperands, linear_varsTypes, linear_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(linear_step_varsOperands, odsBuildableType0, linear_step_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(schedule_chunk_varOperands, schedule_chunk_varTypes, schedule_chunk_varOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WsLoopOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || getLinearVars().size() || getLinearStepVars().size()) {
  _odsPrinter << ' ' << "linear";
  _odsPrinter << "(";
  printLinearClause(_odsPrinter, *this, getLinearVars(), getLinearVars().getTypes(), getLinearStepVars());
  _odsPrinter << ")";
  }
  if (false || getScheduleValAttr() || getScheduleModifierAttr() || getSimdModifierAttr() || getScheduleChunkVar()) {
  _odsPrinter << ' ' << "schedule";
  _odsPrinter << "(";
  printScheduleClause(_odsPrinter, *this, getScheduleValAttr(), getScheduleModifierAttr(), getSimdModifierAttr(), getScheduleChunkVar(), (getScheduleChunkVar() ? getScheduleChunkVar().getType() : ::mlir::Type()));
  _odsPrinter << ")";
  }
  if (false || getNowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  if (false || getOrderedValAttr()) {
  _odsPrinter << ' ' << "ordered";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getOrderedValAttr());
  _odsPrinter << ")";
  }
  if (false || getOrderValAttr()) {
  _odsPrinter << ' ' << "order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, getOrderValAttr());
  _odsPrinter << ")";
  }
  if (false || getReductionVars().size() || getReductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, getReductionVars(), getReductionVars().getTypes(), getReductionsAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "for";
  _odsPrinter << ' ';
  printLoopControl(_odsPrinter, *this, getRegion(), getLowerBound(), getUpperBound(), getStep(), getStep().getTypes(), getInclusiveAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("schedule_val");
  elidedAttrs.push_back("schedule_modifier");
  elidedAttrs.push_back("simd_modifier");
  elidedAttrs.push_back("nowait");
  elidedAttrs.push_back("ordered_val");
  elidedAttrs.push_back("order_val");
  elidedAttrs.push_back("reductions");
  elidedAttrs.push_back("inclusive");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSimdModifierAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("simd_modifier");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNowaitAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nowait");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInclusiveAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inclusive");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
YieldOpGenericAdaptorBase::YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.yield", odsAttrs.getContext());
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

::mlir::Operation::operand_range YieldOp::getResults() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::getResultsMutable() {
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
 build(odsBuilder, odsState, {}); 
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> resultsOperands;
  ::llvm::SMLoc resultsOperandsLoc;
  (void)resultsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  resultsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(resultsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(resultsOperands, resultsTypes, resultsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getResults().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getResults();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

