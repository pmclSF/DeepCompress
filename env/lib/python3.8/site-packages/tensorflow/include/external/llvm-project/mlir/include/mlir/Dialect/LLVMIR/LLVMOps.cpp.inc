/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::LLVM::AShrOp,
::mlir::LLVM::AccessGroupMetadataOp,
::mlir::LLVM::AddOp,
::mlir::LLVM::AddrSpaceCastOp,
::mlir::LLVM::AddressOfOp,
::mlir::LLVM::AliasScopeDomainMetadataOp,
::mlir::LLVM::AliasScopeMetadataOp,
::mlir::LLVM::AllocaOp,
::mlir::LLVM::AndOp,
::mlir::LLVM::AtomicCmpXchgOp,
::mlir::LLVM::AtomicRMWOp,
::mlir::LLVM::BitcastOp,
::mlir::LLVM::BrOp,
::mlir::LLVM::CallOp,
::mlir::LLVM::CondBrOp,
::mlir::LLVM::ConstantOp,
::mlir::LLVM::ExtractElementOp,
::mlir::LLVM::ExtractValueOp,
::mlir::LLVM::FAddOp,
::mlir::LLVM::FCmpOp,
::mlir::LLVM::FDivOp,
::mlir::LLVM::FMulOp,
::mlir::LLVM::FNegOp,
::mlir::LLVM::FPExtOp,
::mlir::LLVM::FPToSIOp,
::mlir::LLVM::FPToUIOp,
::mlir::LLVM::FPTruncOp,
::mlir::LLVM::FRemOp,
::mlir::LLVM::FSubOp,
::mlir::LLVM::FenceOp,
::mlir::LLVM::FreezeOp,
::mlir::LLVM::GEPOp,
::mlir::LLVM::GlobalCtorsOp,
::mlir::LLVM::GlobalDtorsOp,
::mlir::LLVM::GlobalOp,
::mlir::LLVM::ICmpOp,
::mlir::LLVM::InlineAsmOp,
::mlir::LLVM::InsertElementOp,
::mlir::LLVM::InsertValueOp,
::mlir::LLVM::IntToPtrOp,
::mlir::LLVM::InvokeOp,
::mlir::LLVM::LLVMFuncOp,
::mlir::LLVM::LShrOp,
::mlir::LLVM::LandingpadOp,
::mlir::LLVM::LoadOp,
::mlir::LLVM::MetadataOp,
::mlir::LLVM::MulOp,
::mlir::LLVM::NullOp,
::mlir::LLVM::OrOp,
::mlir::LLVM::PoisonOp,
::mlir::LLVM::PtrToIntOp,
::mlir::LLVM::ResumeOp,
::mlir::LLVM::ReturnOp,
::mlir::LLVM::SDivOp,
::mlir::LLVM::SExtOp,
::mlir::LLVM::SIToFPOp,
::mlir::LLVM::SRemOp,
::mlir::LLVM::SelectOp,
::mlir::LLVM::ShlOp,
::mlir::LLVM::ShuffleVectorOp,
::mlir::LLVM::StoreOp,
::mlir::LLVM::SubOp,
::mlir::LLVM::SwitchOp,
::mlir::LLVM::TBAARootMetadataOp,
::mlir::LLVM::TBAATagOp,
::mlir::LLVM::TBAATypeDescriptorOp,
::mlir::LLVM::TruncOp,
::mlir::LLVM::UDivOp,
::mlir::LLVM::UIToFPOp,
::mlir::LLVM::URemOp,
::mlir::LLVM::UndefOp,
::mlir::LLVM::UnreachableOp,
::mlir::LLVM::XOrOp,
::mlir::LLVM::ZExtOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or LLVM dialect-compatible vector of integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::LLVM::LLVMPointerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMPointerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || (((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::IntegerType>())) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMPointerType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to integer or LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::LLVM::LLVMPointerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMStructType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM structure type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || (((::mlir::LLVM::isCompatibleFloatingPointType(type.cast<::mlir::LLVM::LLVMPointerType>().getElementType()))) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::IntegerType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to floating point LLVM type or integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || ((type.isa<::mlir::IntegerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be floating point LLVM type or integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleOuterType(type))) && (!((type.isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM-compatible non-aggregate type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleOuterType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleVectorType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM aggregate type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType(type))))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger(1))) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isSignlessInteger(1)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::IntegerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>())))) || (((type.isa<::mlir::LLVM::LLVMPointerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::LLVM::LLVMPointerType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or LLVM dialect-compatible vector of integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps17(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleOuterType(type))) && ((!type.isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be primitive LLVM type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps18(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || (((((::mlir::LLVM::isCompatibleOuterType(type.cast<::mlir::LLVM::LLVMPointerType>().getElementType()))) && ((!type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>()))) && (!(((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().cast<::mlir::LLVM::LLVMStructType>().isOpaque()))))) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::PointerElementTypeInterface>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to LLVM type with size, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps19(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::mlir::LLVM::isCompatibleOuterType(type))) && ((!type.isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>()))) && (!(((type.isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMStructType>().isOpaque()))))) || ((type.isa<::mlir::LLVM::PointerElementTypeInterface>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM type with size, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps20(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::AtomicOrderingAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Atomic ordering for LLVM's memory model";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::AtomicBinOpAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: llvm.atomicrmw binary operations";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::LoopAnnotationAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::FastmathFlagsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM fastmath flags";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ElementsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: constant vector/tensor attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseI64ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i64 dense array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::FCmpPredicateAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: llvm.fcmp comparison predicate";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseI32ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i32 dense array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps17(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::LinkageAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM Linkage specification";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps18(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!attr.cast<::mlir::IntegerAttr>().getValue().isNegative())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps19(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::UnnamedAddrAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM GlobalValue UnnamedAddr";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps20(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::VisibilityAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM GlobalValue Visibility";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps21(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::ICmpPredicateAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: lvm.icmp comparison predicate";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps22(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::AsmDialectAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ATT (0) or Intel (1) asm dialect";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps23(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps24(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::LLVM::LLVMFunctionType>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::LLVM::LLVMFunctionType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type attribute of LLVM function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps25(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::CConvAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM Calling Convention specification";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps26(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Array of dictionary attributes";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps27(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::MemoryEffectsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_LLVMOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_LLVMOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AShrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AShrOpGenericAdaptorBase::AShrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.ashr", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AShrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AShrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AShrOpAdaptor::AShrOpAdaptor(AShrOp op) : AShrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value AShrOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AShrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AShrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AShrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AShrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AShrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AShrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult AShrOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AShrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AShrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AShrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AShrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AccessGroupMetadataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AccessGroupMetadataOpGenericAdaptorBase::AccessGroupMetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.access_group", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AccessGroupMetadataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AccessGroupMetadataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AccessGroupMetadataOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AccessGroupMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef AccessGroupMetadataOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

} // namespace detail
AccessGroupMetadataOpAdaptor::AccessGroupMetadataOpAdaptor(AccessGroupMetadataOp op) : AccessGroupMetadataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AccessGroupMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.access_group' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == AccessGroupMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.access_group' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AccessGroupMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AccessGroupMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AccessGroupMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AccessGroupMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AccessGroupMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AccessGroupMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

void AccessGroupMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void AccessGroupMetadataOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AccessGroupMetadataOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult AccessGroupMetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AccessGroupMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AccessGroupMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AccessGroupMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AddOpGenericAdaptorBase::AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.add", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AddOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AddOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AddOpAdaptor::AddOpAdaptor(AddOp op) : AddOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value AddOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult AddOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddrSpaceCastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AddrSpaceCastOpGenericAdaptorBase::AddrSpaceCastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.addrspacecast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AddrSpaceCastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AddrSpaceCastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AddrSpaceCastOpAdaptor::AddrSpaceCastOpAdaptor(AddrSpaceCastOp op) : AddrSpaceCastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AddrSpaceCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddrSpaceCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddrSpaceCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange AddrSpaceCastOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddrSpaceCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddrSpaceCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddrSpaceCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddrSpaceCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AddrSpaceCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AddrSpaceCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AddrSpaceCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AddrSpaceCastOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddressOfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AddressOfOpGenericAdaptorBase::AddressOfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.addressof", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AddressOfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AddressOfOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr AddressOfOpGenericAdaptorBase::getGlobalNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AddressOfOp::getGlobalNameAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::StringRef AddressOfOpGenericAdaptorBase::getGlobalName() {
  auto attr = getGlobalNameAttr();
  return attr.getValue();
}

} // namespace detail
AddressOfOpAdaptor::AddressOfOpAdaptor(AddressOfOp op) : AddressOfOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AddressOfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.addressof' op ""requires attribute 'global_name'");
    if (namedAttrIt->getName() == AddressOfOp::getGlobalNameAttrName(*odsOpName)) {
      tblgen_global_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_global_name && !((tblgen_global_name.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.mlir.addressof' op ""attribute 'global_name' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddressOfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddressOfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AddressOfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddressOfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> AddressOfOp::getRes() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSResults(0).begin());
}

::mlir::FlatSymbolRefAttr AddressOfOp::getGlobalNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGlobalNameAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef AddressOfOp::getGlobalName() {
  auto attr = getGlobalNameAttr();
  return attr.getValue();
}

void AddressOfOp::setGlobalNameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getGlobalNameAttrName(), attr);
}

void AddressOfOp::setGlobalName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getGlobalNameAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, GlobalOp global, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState,
            LLVM::LLVMPointerType::get(global.getType(), global.getAddrSpace()),
            global.getSymName());
      odsState.addAttributes(attrs);
    
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFuncOp func, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState,
            LLVM::LLVMPointerType::get(func.getFunctionType()), func.getName());
      odsState.addAttributes(attrs);
    
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::FlatSymbolRefAttr global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), global_name);
  odsState.addTypes(res);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), global_name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::llvm::StringRef global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), global_name));
  odsState.addTypes(res);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), global_name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddressOfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddressOfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global_name'");
    if (namedAttrIt->getName() == getGlobalNameAttrName()) {
      tblgen_global_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_global_name, "global_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddressOfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AddressOfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr global_nameAttr;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  if (parser.parseCustomAttributeWithFallback(global_nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "global_name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void AddressOfOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getGlobalNameAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("global_name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getRes().getType();
}

void AddressOfOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AddressOfOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AliasScopeDomainMetadataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AliasScopeDomainMetadataOpGenericAdaptorBase::AliasScopeDomainMetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.alias_scope_domain", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AliasScopeDomainMetadataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AliasScopeDomainMetadataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AliasScopeDomainMetadataOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AliasScopeDomainMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef AliasScopeDomainMetadataOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeDomainMetadataOpGenericAdaptorBase::getDescriptionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AliasScopeDomainMetadataOp::getDescriptionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > AliasScopeDomainMetadataOpGenericAdaptorBase::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
AliasScopeDomainMetadataOpAdaptor::AliasScopeDomainMetadataOpAdaptor(AliasScopeDomainMetadataOp op) : AliasScopeDomainMetadataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AliasScopeDomainMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.alias_scope_domain' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == AliasScopeDomainMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AliasScopeDomainMetadataOp::getDescriptionAttrName(*odsOpName)) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope_domain' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_description && !((tblgen_description.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope_domain' op ""attribute 'description' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AliasScopeDomainMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AliasScopeDomainMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AliasScopeDomainMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AliasScopeDomainMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AliasScopeDomainMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AliasScopeDomainMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeDomainMetadataOp::getDescriptionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDescriptionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > AliasScopeDomainMetadataOp::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void AliasScopeDomainMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void AliasScopeDomainMetadataOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void AliasScopeDomainMetadataOp::setDescriptionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDescriptionAttrName(), attr);
}

void AliasScopeDomainMetadataOp::setDescription(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDescriptionAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getDescriptionAttrName());
}

::mlir::Attribute AliasScopeDomainMetadataOp::removeDescriptionAttr() {
  return (*this)->removeAttr(getDescriptionAttrName());
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AliasScopeDomainMetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDescriptionAttrName()) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_description, "description")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult AliasScopeDomainMetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AliasScopeDomainMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AliasScopeDomainMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AliasScopeDomainMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AliasScopeMetadataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AliasScopeMetadataOpGenericAdaptorBase::AliasScopeMetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.alias_scope", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AliasScopeMetadataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AliasScopeMetadataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AliasScopeMetadataOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AliasScopeMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef AliasScopeMetadataOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr AliasScopeMetadataOpGenericAdaptorBase::getDomainAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AliasScopeMetadataOp::getDomainAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::StringRef AliasScopeMetadataOpGenericAdaptorBase::getDomain() {
  auto attr = getDomainAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeMetadataOpGenericAdaptorBase::getDescriptionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AliasScopeMetadataOp::getDescriptionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > AliasScopeMetadataOpGenericAdaptorBase::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
AliasScopeMetadataOpAdaptor::AliasScopeMetadataOpAdaptor(AliasScopeMetadataOp op) : AliasScopeMetadataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AliasScopeMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_domain;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.alias_scope' op ""requires attribute 'domain'");
    if (namedAttrIt->getName() == AliasScopeMetadataOp::getDomainAttrName(*odsOpName)) {
      tblgen_domain = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AliasScopeMetadataOp::getDescriptionAttrName(*odsOpName)) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.alias_scope' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == AliasScopeMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_domain && !((tblgen_domain.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.alias_scope' op ""attribute 'domain' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_description && !((tblgen_description.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope' op ""attribute 'description' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AliasScopeMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AliasScopeMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AliasScopeMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AliasScopeMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AliasScopeMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AliasScopeMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr AliasScopeMetadataOp::getDomainAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDomainAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef AliasScopeMetadataOp::getDomain() {
  auto attr = getDomainAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeMetadataOp::getDescriptionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDescriptionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > AliasScopeMetadataOp::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void AliasScopeMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void AliasScopeMetadataOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void AliasScopeMetadataOp::setDomainAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getDomainAttrName(), attr);
}

void AliasScopeMetadataOp::setDomain(::llvm::StringRef attrValue) {
  (*this)->setAttr(getDomainAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AliasScopeMetadataOp::setDescriptionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDescriptionAttrName(), attr);
}

void AliasScopeMetadataOp::setDescription(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDescriptionAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getDescriptionAttrName());
}

::mlir::Attribute AliasScopeMetadataOp::removeDescriptionAttr() {
  return (*this)->removeAttr(getDescriptionAttrName());
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::FlatSymbolRefAttr domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getDomainAttrName(odsState.name), domain);
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::FlatSymbolRefAttr domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getDomainAttrName(odsState.name), domain);
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::llvm::StringRef domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getDomainAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), domain));
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::llvm::StringRef domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getDomainAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), domain));
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AliasScopeMetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_domain;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'domain'");
    if (namedAttrIt->getName() == getDomainAttrName()) {
      tblgen_domain = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDescriptionAttrName()) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_domain, "domain")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_description, "description")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult AliasScopeMetadataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AliasScopeMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AliasScopeMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AliasScopeMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AllocaOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AllocaOpGenericAdaptorBase::AllocaOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.alloca", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AllocaOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AllocaOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AllocaOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocaOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> AllocaOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::TypeAttr AllocaOpGenericAdaptorBase::getElemTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocaOp::getElemTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::std::optional<::mlir::Type> AllocaOpGenericAdaptorBase::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::std::optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::std::nullopt);
}

::mlir::UnitAttr AllocaOpGenericAdaptorBase::getInallocaAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocaOp::getInallocaAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool AllocaOpGenericAdaptorBase::getInalloca() {
  auto attr = getInallocaAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
AllocaOpAdaptor::AllocaOpAdaptor(AllocaOp op) : AllocaOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AllocaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_elem_type;
  ::mlir::Attribute tblgen_inalloca;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AllocaOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AllocaOp::getElemTypeAttrName(*odsOpName)) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AllocaOp::getInallocaAttrName(*odsOpName)) {
      tblgen_inalloca = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.alloca' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_elem_type && !(((tblgen_elem_type.isa<::mlir::TypeAttr>())) && ((tblgen_elem_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'llvm.alloca' op ""attribute 'elem_type' failed to satisfy constraint: any type attribute");

  if (tblgen_inalloca && !((tblgen_inalloca.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.alloca' op ""attribute 'inalloca' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AllocaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AllocaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> AllocaOp::getArraySize() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange AllocaOp::getArraySizeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AllocaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> AllocaOp::getRes() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr AllocaOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> AllocaOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::TypeAttr AllocaOp::getElemTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElemTypeAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::std::optional<::mlir::Type> AllocaOp::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::std::optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::std::nullopt);
}

::mlir::UnitAttr AllocaOp::getInallocaAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getInallocaAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool AllocaOp::getInalloca() {
  auto attr = getInallocaAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void AllocaOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AllocaOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

void AllocaOp::setElemTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getElemTypeAttrName(), attr);
}

void AllocaOp::setElemType(::std::optional<::mlir::Type> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getElemTypeAttrName(), ::mlir::TypeAttr::get(*attrValue));
    (*this)->removeAttr(getElemTypeAttrName());
}

void AllocaOp::setInallocaAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInallocaAttrName(), attr);
}

void AllocaOp::setInalloca(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInallocaAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInallocaAttrName());
}

::mlir::Attribute AllocaOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute AllocaOp::removeElemTypeAttr() {
  return (*this)->removeAttr(getElemTypeAttrName());
}

::mlir::Attribute AllocaOp::removeInallocaAttr() {
  return (*this)->removeAttr(getInallocaAttrName());
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value arraySize, unsigned alignment) {
      assert(!resultType.cast<LLVMPointerType>().isOpaque() &&
             "pass the allocated type explicitly if opaque pointers are used");
      if (alignment == 0)
        return build(odsBuilder, odsState, resultType, arraySize, IntegerAttr(),
                     TypeAttr(), false);
      build(odsBuilder, odsState, resultType, arraySize,
        odsBuilder.getI64IntegerAttr(alignment), TypeAttr(), false);
    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Type elementType, Value arraySize, unsigned alignment) {
      TypeAttr elemTypeAttr =
          resultType.cast<LLVMPointerType>().isOpaque() ?
          TypeAttr::get(elementType) : TypeAttr();
      build(odsBuilder, odsState, resultType, arraySize,
            alignment == 0 ? IntegerAttr()
                           : odsBuilder.getI64IntegerAttr(alignment),
            elemTypeAttr, false);

    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type, /*optional*/::mlir::UnitAttr inalloca) {
  odsState.addOperands(arraySize);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (elem_type) {
    odsState.addAttribute(getElemTypeAttrName(odsState.name), elem_type);
  }
  if (inalloca) {
    odsState.addAttribute(getInallocaAttrName(odsState.name), inalloca);
  }
  odsState.addTypes(res);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type, /*optional*/::mlir::UnitAttr inalloca) {
  odsState.addOperands(arraySize);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (elem_type) {
    odsState.addAttribute(getElemTypeAttrName(odsState.name), elem_type);
  }
  if (inalloca) {
    odsState.addAttribute(getInallocaAttrName(odsState.name), inalloca);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type, /*optional*/bool inalloca) {
  odsState.addOperands(arraySize);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (elem_type) {
    odsState.addAttribute(getElemTypeAttrName(odsState.name), elem_type);
  }
  if (inalloca) {
    odsState.addAttribute(getInallocaAttrName(odsState.name), ((inalloca) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(res);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type, /*optional*/bool inalloca) {
  odsState.addOperands(arraySize);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (elem_type) {
    odsState.addAttribute(getElemTypeAttrName(odsState.name), elem_type);
  }
  if (inalloca) {
    odsState.addAttribute(getInallocaAttrName(odsState.name), ((inalloca) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_elem_type;
  ::mlir::Attribute tblgen_inalloca;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getElemTypeAttrName()) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getInallocaAttrName()) {
      tblgen_inalloca = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps3(*this, tblgen_elem_type, "elem_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_inalloca, "inalloca")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocaOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AllocaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AllocaOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AndOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AndOpGenericAdaptorBase::AndOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.and", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AndOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AndOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AndOpAdaptor::AndOpAdaptor(AndOp op) : AndOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value AndOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AndOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AndOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AndOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AndOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult AndOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AndOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AndOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AtomicCmpXchgOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicCmpXchgOpGenericAdaptorBase::AtomicCmpXchgOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.cmpxchg", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicCmpXchgOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicCmpXchgOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOpGenericAdaptorBase::getSuccessOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AtomicCmpXchgOp::getSuccessOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOpGenericAdaptorBase::getSuccessOrdering() {
  auto attr = getSuccessOrderingAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOpGenericAdaptorBase::getFailureOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AtomicCmpXchgOp::getFailureOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOpGenericAdaptorBase::getFailureOrdering() {
  auto attr = getFailureOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr AtomicCmpXchgOpGenericAdaptorBase::getSyncscopeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicCmpXchgOp::getSyncscopeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > AtomicCmpXchgOpGenericAdaptorBase::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr AtomicCmpXchgOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AtomicCmpXchgOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> AtomicCmpXchgOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr AtomicCmpXchgOpGenericAdaptorBase::getWeakAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicCmpXchgOp::getWeakAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool AtomicCmpXchgOpGenericAdaptorBase::getWeak() {
  auto attr = getWeakAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr AtomicCmpXchgOpGenericAdaptorBase::getVolatile_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicCmpXchgOp::getVolatile_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool AtomicCmpXchgOpGenericAdaptorBase::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr AtomicCmpXchgOpGenericAdaptorBase::getAccessGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AtomicCmpXchgOp::getAccessGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOpGenericAdaptorBase::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicCmpXchgOpGenericAdaptorBase::getAliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AtomicCmpXchgOp::getAliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOpGenericAdaptorBase::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicCmpXchgOpGenericAdaptorBase::getNoaliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, AtomicCmpXchgOp::getNoaliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOpGenericAdaptorBase::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicCmpXchgOpGenericAdaptorBase::getTbaaAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicCmpXchgOp::getTbaaAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOpGenericAdaptorBase::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
AtomicCmpXchgOpAdaptor::AtomicCmpXchgOpAdaptor(AtomicCmpXchgOp op) : AtomicCmpXchgOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicCmpXchgOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_failure_ordering;
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.cmpxchg' op ""requires attribute 'failure_ordering'");
    if (namedAttrIt->getName() == AtomicCmpXchgOp::getFailureOrderingAttrName(*odsOpName)) {
      tblgen_failure_ordering = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getAccessGroupsAttrName(*odsOpName)) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getAliasScopesAttrName(*odsOpName)) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_success_ordering;
  ::mlir::Attribute tblgen_noalias_scopes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.cmpxchg' op ""requires attribute 'success_ordering'");
    if (namedAttrIt->getName() == AtomicCmpXchgOp::getSuccessOrderingAttrName(*odsOpName)) {
      tblgen_success_ordering = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getNoaliasScopesAttrName(*odsOpName)) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  ::mlir::Attribute tblgen_weak;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getSyncscopeAttrName(*odsOpName)) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getTbaaAttrName(*odsOpName)) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getVolatile_AttrName(*odsOpName)) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCmpXchgOp::getWeakAttrName(*odsOpName)) {
      tblgen_weak = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_success_ordering && !((tblgen_success_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'success_ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_failure_ordering && !((tblgen_failure_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'failure_ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_syncscope && !((tblgen_syncscope.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_weak && !((tblgen_weak.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'weak' failed to satisfy constraint: unit attribute");

  if (tblgen_volatile_ && !((tblgen_volatile_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");

  if (tblgen_access_groups && !(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alias_scopes && !(((tblgen_alias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_noalias_scopes && !(((tblgen_noalias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_noalias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_tbaa && !(((tblgen_tbaa.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_tbaa.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'tbaa' failed to satisfy constraint: symbol ref array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicCmpXchgOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicCmpXchgOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> AtomicCmpXchgOp::getPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSOperands(0).begin());
}

::mlir::Value AtomicCmpXchgOp::getCmp() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value AtomicCmpXchgOp::getVal() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
}

::mlir::MutableOperandRange AtomicCmpXchgOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicCmpXchgOp::getCmpMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicCmpXchgOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicCmpXchgOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicCmpXchgOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOp::getSuccessOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSuccessOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOp::getSuccessOrdering() {
  auto attr = getSuccessOrderingAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOp::getFailureOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFailureOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOp::getFailureOrdering() {
  auto attr = getFailureOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr AtomicCmpXchgOp::getSyncscopeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSyncscopeAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > AtomicCmpXchgOp::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr AtomicCmpXchgOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> AtomicCmpXchgOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr AtomicCmpXchgOp::getWeakAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getWeakAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool AtomicCmpXchgOp::getWeak() {
  auto attr = getWeakAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr AtomicCmpXchgOp::getVolatile_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getVolatile_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool AtomicCmpXchgOp::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr AtomicCmpXchgOp::getAccessGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAccessGroupsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOp::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicCmpXchgOp::getAliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOp::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicCmpXchgOp::getNoaliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getNoaliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOp::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicCmpXchgOp::getTbaaAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getTbaaAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicCmpXchgOp::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void AtomicCmpXchgOp::setSuccessOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getSuccessOrderingAttrName(), attr);
}

void AtomicCmpXchgOp::setSuccessOrdering(::mlir::LLVM::AtomicOrdering attrValue) {
  (*this)->setAttr(getSuccessOrderingAttrName(), ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AtomicCmpXchgOp::setFailureOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getFailureOrderingAttrName(), attr);
}

void AtomicCmpXchgOp::setFailureOrdering(::mlir::LLVM::AtomicOrdering attrValue) {
  (*this)->setAttr(getFailureOrderingAttrName(), ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AtomicCmpXchgOp::setSyncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSyncscopeAttrName(), attr);
}

void AtomicCmpXchgOp::setSyncscope(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSyncscopeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSyncscopeAttrName());
}

void AtomicCmpXchgOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AtomicCmpXchgOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

void AtomicCmpXchgOp::setWeakAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getWeakAttrName(), attr);
}

void AtomicCmpXchgOp::setWeak(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getWeakAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getWeakAttrName());
}

void AtomicCmpXchgOp::setVolatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getVolatile_AttrName(), attr);
}

void AtomicCmpXchgOp::setVolatile_(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getVolatile_AttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getVolatile_AttrName());
}

void AtomicCmpXchgOp::setAccessGroupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAccessGroupsAttrName(), attr);
}

void AtomicCmpXchgOp::setAliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAliasScopesAttrName(), attr);
}

void AtomicCmpXchgOp::setNoaliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNoaliasScopesAttrName(), attr);
}

void AtomicCmpXchgOp::setTbaaAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTbaaAttrName(), attr);
}

::mlir::Attribute AtomicCmpXchgOp::removeSyncscopeAttr() {
  return (*this)->removeAttr(getSyncscopeAttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeWeakAttr() {
  return (*this)->removeAttr(getWeakAttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeVolatile_Attr() {
  return (*this)->removeAttr(getVolatile_AttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeAccessGroupsAttr() {
  return (*this)->removeAttr(getAccessGroupsAttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeAliasScopesAttr() {
  return (*this)->removeAttr(getAliasScopesAttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeNoaliasScopesAttr() {
  return (*this)->removeAttr(getNoaliasScopesAttrName());
}

::mlir::Attribute AtomicCmpXchgOp::removeTbaaAttr() {
  return (*this)->removeAttr(getTbaaAttrName());
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr weak, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), success_ordering);
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), failure_ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (weak) {
    odsState.addAttribute(getWeakAttrName(odsState.name), weak);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  odsState.addTypes(res);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr weak, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), success_ordering);
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), failure_ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (weak) {
    odsState.addAttribute(getWeakAttrName(odsState.name), weak);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtomicCmpXchgOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr weak, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), success_ordering);
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), failure_ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (weak) {
    odsState.addAttribute(getWeakAttrName(odsState.name), weak);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool weak, /*optional*/bool volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (weak) {
    odsState.addAttribute(getWeakAttrName(odsState.name), ((weak) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  odsState.addTypes(res);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool weak, /*optional*/bool volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (weak) {
    odsState.addAttribute(getWeakAttrName(odsState.name), ((weak) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtomicCmpXchgOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool weak, /*optional*/bool volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (weak) {
    odsState.addAttribute(getWeakAttrName(odsState.name), ((weak) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AtomicCmpXchgOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AtomicCmpXchgOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_failure_ordering;
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'failure_ordering'");
    if (namedAttrIt->getName() == getFailureOrderingAttrName()) {
      tblgen_failure_ordering = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAccessGroupsAttrName()) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAliasScopesAttrName()) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_success_ordering;
  ::mlir::Attribute tblgen_noalias_scopes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'success_ordering'");
    if (namedAttrIt->getName() == getSuccessOrderingAttrName()) {
      tblgen_success_ordering = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNoaliasScopesAttrName()) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  ::mlir::Attribute tblgen_weak;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSyncscopeAttrName()) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTbaaAttrName()) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVolatile_AttrName()) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getWeakAttrName()) {
      tblgen_weak = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_success_ordering, "success_ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_failure_ordering, "failure_ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_syncscope, "syncscope")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_weak, "weak")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_volatile_, "volatile_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_access_groups, "access_groups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_alias_scopes, "alias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_noalias_scopes, "noalias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_tbaa, "tbaa")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(2).begin()).getType(), (*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that operand #1 and operand #2 have the same type");
  if (!((std::equal_to<>()(getValAndBoolStructType((*this->getODSOperands(2).begin()).getType()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result #0 has an LLVM struct type consisting of the type of operand #2 and a bool");
  return ::mlir::success();
}

::mlir::LogicalResult AtomicCmpXchgOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult AtomicCmpXchgOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = getValAndBoolStructType(operands[2].getType());
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AtomicCmpXchgOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cmpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cmpOperands(cmpRawOperands);  ::llvm::SMLoc cmpOperandsLoc;
  (void)cmpOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valOperands(valRawOperands);  ::llvm::SMLoc valOperandsLoc;
  (void)valOperandsLoc;
  ::mlir::StringAttr syncscopeAttr;
  ::mlir::LLVM::AtomicOrderingAttr success_orderingAttr;
  ::mlir::LLVM::AtomicOrderingAttr failure_orderingAttr;
  ::mlir::Type ptrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> ptrTypes(ptrRawTypes);
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);
  if (::mlir::succeeded(parser.parseOptionalKeyword("weak"))) {
    result.addAttribute("weak", parser.getBuilder().getUnitAttr());
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("volatile"))) {
    result.addAttribute("volatile_", parser.getBuilder().getUnitAttr());
  }

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  cmpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cmpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("syncscope"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(syncscopeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "syncscope",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"not_atomic","unordered","monotonic","acquire","release","acq_rel","seq_cst"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "success_ordering", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'success_ordering' [not_atomic, unordered, monotonic, acquire, release, acq_rel, seq_cst]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicOrdering(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "success_ordering attribute specification: \"" << attrStr << '"';;

      success_orderingAttr = ::mlir::LLVM::AtomicOrderingAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("success_ordering", success_orderingAttr);
    }
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"not_atomic","unordered","monotonic","acquire","release","acq_rel","seq_cst"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "failure_ordering", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'failure_ordering' [not_atomic, unordered, monotonic, acquire, release, acq_rel, seq_cst]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicOrdering(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "failure_ordering attribute specification: \"" << attrStr << '"';;

      failure_orderingAttr = ::mlir::LLVM::AtomicOrderingAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("failure_ordering", failure_orderingAttr);
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(ptrRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valRawTypes[0] = type;
  }
  for (::mlir::Type type : valTypes) {
    (void)type;
    if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::LLVM::LLVMPointerType>())))) {
      return parser.emitError(parser.getNameLoc()) << "'val' must be integer or LLVM pointer type, but got " << type;
    }
  }
  result.addTypes(getValAndBoolStructType(valTypes[0]));
  if (parser.resolveOperands(ptrOperands, ptrTypes, ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cmpOperands, valTypes[0], cmpOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valOperands, valTypes, valOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicCmpXchgOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("weak")) {
    _odsPrinter << ' ' << "weak";
  }
  if ((*this)->getAttr("volatile_")) {
    _odsPrinter << ' ' << "volatile";
  }
  _odsPrinter << ' ';
  _odsPrinter << getPtr();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCmp();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getVal();
  if ((*this)->getAttr("syncscope")) {
    _odsPrinter << ' ' << "syncscope";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getSyncscopeAttr());
    _odsPrinter << ")";
  }
  _odsPrinter << ' ';

  {
    auto caseValue = getSuccessOrdering();
    auto caseValueStr = stringifyAtomicOrdering(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ' ';

  {
    auto caseValue = getFailureOrdering();
    auto caseValueStr = stringifyAtomicOrdering(caseValue);
    _odsPrinter << caseValueStr;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("weak");
  elidedAttrs.push_back("volatile_");
  elidedAttrs.push_back("syncscope");
  elidedAttrs.push_back("success_ordering");
  elidedAttrs.push_back("failure_ordering");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWeakAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("weak");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getVolatile_Attr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("volatile_");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getPtr().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getVal().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AtomicCmpXchgOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AtomicRMWOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicRMWOpGenericAdaptorBase::AtomicRMWOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.atomicrmw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicRMWOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicRMWOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicBinOpAttr AtomicRMWOpGenericAdaptorBase::getBinOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AtomicRMWOp::getBinOpAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicBinOpAttr>();
  return attr;
}

::mlir::LLVM::AtomicBinOp AtomicRMWOpGenericAdaptorBase::getBinOp() {
  auto attr = getBinOpAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicRMWOpGenericAdaptorBase::getOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AtomicRMWOp::getOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrdering AtomicRMWOpGenericAdaptorBase::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr AtomicRMWOpGenericAdaptorBase::getSyncscopeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicRMWOp::getSyncscopeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > AtomicRMWOpGenericAdaptorBase::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr AtomicRMWOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AtomicRMWOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> AtomicRMWOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr AtomicRMWOpGenericAdaptorBase::getVolatile_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicRMWOp::getVolatile_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool AtomicRMWOpGenericAdaptorBase::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr AtomicRMWOpGenericAdaptorBase::getAccessGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AtomicRMWOp::getAccessGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOpGenericAdaptorBase::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicRMWOpGenericAdaptorBase::getAliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AtomicRMWOp::getAliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOpGenericAdaptorBase::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicRMWOpGenericAdaptorBase::getNoaliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, AtomicRMWOp::getNoaliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOpGenericAdaptorBase::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicRMWOpGenericAdaptorBase::getTbaaAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AtomicRMWOp::getTbaaAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOpGenericAdaptorBase::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(AtomicRMWOp op) : AtomicRMWOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_bin_op;
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.atomicrmw' op ""requires attribute 'bin_op'");
    if (namedAttrIt->getName() == AtomicRMWOp::getBinOpAttrName(*odsOpName)) {
      tblgen_bin_op = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getAccessGroupsAttrName(*odsOpName)) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getAliasScopesAttrName(*odsOpName)) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_ordering;
  ::mlir::Attribute tblgen_noalias_scopes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.atomicrmw' op ""requires attribute 'ordering'");
    if (namedAttrIt->getName() == AtomicRMWOp::getOrderingAttrName(*odsOpName)) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getNoaliasScopesAttrName(*odsOpName)) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getSyncscopeAttrName(*odsOpName)) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getTbaaAttrName(*odsOpName)) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicRMWOp::getVolatile_AttrName(*odsOpName)) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_bin_op && !((tblgen_bin_op.isa<::mlir::LLVM::AtomicBinOpAttr>())))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'bin_op' failed to satisfy constraint: llvm.atomicrmw binary operations");

  if (tblgen_ordering && !((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_syncscope && !((tblgen_syncscope.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_volatile_ && !((tblgen_volatile_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");

  if (tblgen_access_groups && !(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alias_scopes && !(((tblgen_alias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_noalias_scopes && !(((tblgen_noalias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_noalias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_tbaa && !(((tblgen_tbaa.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_tbaa.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'tbaa' failed to satisfy constraint: symbol ref array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> AtomicRMWOp::getPtr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSOperands(0).begin());
}

::mlir::Value AtomicRMWOp::getVal() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AtomicRMWOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicRMWOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::AtomicBinOpAttr AtomicRMWOp::getBinOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBinOpAttrName()).cast<::mlir::LLVM::AtomicBinOpAttr>();
}

::mlir::LLVM::AtomicBinOp AtomicRMWOp::getBinOp() {
  auto attr = getBinOpAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicRMWOp::getOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicRMWOp::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr AtomicRMWOp::getSyncscopeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSyncscopeAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > AtomicRMWOp::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr AtomicRMWOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> AtomicRMWOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr AtomicRMWOp::getVolatile_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getVolatile_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool AtomicRMWOp::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::ArrayAttr AtomicRMWOp::getAccessGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAccessGroupsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOp::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicRMWOp::getAliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOp::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicRMWOp::getNoaliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getNoaliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOp::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr AtomicRMWOp::getTbaaAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getTbaaAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > AtomicRMWOp::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void AtomicRMWOp::setBinOpAttr(::mlir::LLVM::AtomicBinOpAttr attr) {
  (*this)->setAttr(getBinOpAttrName(), attr);
}

void AtomicRMWOp::setBinOp(::mlir::LLVM::AtomicBinOp attrValue) {
  (*this)->setAttr(getBinOpAttrName(), ::mlir::LLVM::AtomicBinOpAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AtomicRMWOp::setOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getOrderingAttrName(), attr);
}

void AtomicRMWOp::setOrdering(::mlir::LLVM::AtomicOrdering attrValue) {
  (*this)->setAttr(getOrderingAttrName(), ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AtomicRMWOp::setSyncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSyncscopeAttrName(), attr);
}

void AtomicRMWOp::setSyncscope(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSyncscopeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSyncscopeAttrName());
}

void AtomicRMWOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AtomicRMWOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

void AtomicRMWOp::setVolatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getVolatile_AttrName(), attr);
}

void AtomicRMWOp::setVolatile_(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getVolatile_AttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getVolatile_AttrName());
}

void AtomicRMWOp::setAccessGroupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAccessGroupsAttrName(), attr);
}

void AtomicRMWOp::setAliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAliasScopesAttrName(), attr);
}

void AtomicRMWOp::setNoaliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNoaliasScopesAttrName(), attr);
}

void AtomicRMWOp::setTbaaAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTbaaAttrName(), attr);
}

::mlir::Attribute AtomicRMWOp::removeSyncscopeAttr() {
  return (*this)->removeAttr(getSyncscopeAttrName());
}

::mlir::Attribute AtomicRMWOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute AtomicRMWOp::removeVolatile_Attr() {
  return (*this)->removeAttr(getVolatile_AttrName());
}

::mlir::Attribute AtomicRMWOp::removeAccessGroupsAttr() {
  return (*this)->removeAttr(getAccessGroupsAttrName());
}

::mlir::Attribute AtomicRMWOp::removeAliasScopesAttr() {
  return (*this)->removeAttr(getAliasScopesAttrName());
}

::mlir::Attribute AtomicRMWOp::removeNoaliasScopesAttr() {
  return (*this)->removeAttr(getNoaliasScopesAttrName());
}

::mlir::Attribute AtomicRMWOp::removeTbaaAttr() {
  return (*this)->removeAttr(getTbaaAttrName());
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), bin_op);
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  odsState.addTypes(res);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), bin_op);
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtomicRMWOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), bin_op);
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  odsState.addTypes(res);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtomicRMWOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AtomicRMWOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AtomicRMWOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_bin_op;
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'bin_op'");
    if (namedAttrIt->getName() == getBinOpAttrName()) {
      tblgen_bin_op = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAccessGroupsAttrName()) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAliasScopesAttrName()) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_ordering;
  ::mlir::Attribute tblgen_noalias_scopes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ordering'");
    if (namedAttrIt->getName() == getOrderingAttrName()) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNoaliasScopesAttrName()) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSyncscopeAttrName()) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTbaaAttrName()) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVolatile_AttrName()) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps7(*this, tblgen_bin_op, "bin_op")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_ordering, "ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_syncscope, "syncscope")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_volatile_, "volatile_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_access_groups, "access_groups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_alias_scopes, "alias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_noalias_scopes, "noalias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_tbaa, "tbaa")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result #0 and operand #1 have the same type");
  return ::mlir::success();
}

::mlir::LogicalResult AtomicRMWOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult AtomicRMWOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AtomicRMWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::LLVM::AtomicBinOpAttr bin_opAttr;
  ::mlir::OpAsmParser::UnresolvedOperand ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valOperands(valRawOperands);  ::llvm::SMLoc valOperandsLoc;
  (void)valOperandsLoc;
  ::mlir::StringAttr syncscopeAttr;
  ::mlir::LLVM::AtomicOrderingAttr orderingAttr;
  ::mlir::Type ptrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> ptrTypes(ptrRawTypes);
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);
  if (::mlir::succeeded(parser.parseOptionalKeyword("volatile"))) {
    result.addAttribute("volatile_", parser.getBuilder().getUnitAttr());
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"xchg","add","sub","_and","nand","_or","_xor","max","min","umax","umin","fadd","fsub","fmax","fmin","uinc_wrap","udec_wrap"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "bin_op", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'bin_op' [xchg, add, sub, _and, nand, _or, _xor, max, min, umax, umin, fadd, fsub, fmax, fmin, uinc_wrap, udec_wrap]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicBinOp(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "bin_op attribute specification: \"" << attrStr << '"';;

      bin_opAttr = ::mlir::LLVM::AtomicBinOpAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("bin_op", bin_opAttr);
    }
  }

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("syncscope"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(syncscopeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "syncscope",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"not_atomic","unordered","monotonic","acquire","release","acq_rel","seq_cst"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "ordering", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'ordering' [not_atomic, unordered, monotonic, acquire, release, acq_rel, seq_cst]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicOrdering(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "ordering attribute specification: \"" << attrStr << '"';;

      orderingAttr = ::mlir::LLVM::AtomicOrderingAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("ordering", orderingAttr);
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(ptrRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valRawTypes[0] = type;
  }
  for (::mlir::Type type : valTypes) {
    (void)type;
    if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || ((type.isa<::mlir::IntegerType>())))) {
      return parser.emitError(parser.getNameLoc()) << "'val' must be floating point LLVM type or integer, but got " << type;
    }
  }
  result.addTypes(valTypes[0]);
  if (parser.resolveOperands(ptrOperands, ptrTypes, ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valOperands, valTypes, valOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicRMWOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("volatile_")) {
    _odsPrinter << ' ' << "volatile";
  }
  _odsPrinter << ' ';

  {
    auto caseValue = getBinOp();
    auto caseValueStr = stringifyAtomicBinOp(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ' ';
  _odsPrinter << getPtr();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getVal();
  if ((*this)->getAttr("syncscope")) {
    _odsPrinter << ' ' << "syncscope";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getSyncscopeAttr());
    _odsPrinter << ")";
  }
  _odsPrinter << ' ';

  {
    auto caseValue = getOrdering();
    auto caseValueStr = stringifyAtomicOrdering(caseValue);
    _odsPrinter << caseValueStr;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("volatile_");
  elidedAttrs.push_back("bin_op");
  elidedAttrs.push_back("syncscope");
  elidedAttrs.push_back("ordering");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getVolatile_Attr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("volatile_");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getPtr().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getVal().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AtomicRMWOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BitcastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BitcastOpGenericAdaptorBase::BitcastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.bitcast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BitcastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BitcastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BitcastOpAdaptor::BitcastOpAdaptor(BitcastOp op) : BitcastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BitcastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitcastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitcastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange BitcastOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitcastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitcastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitcastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitcastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitcastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BitcastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BitcastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BitcastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::BitcastOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BrOpGenericAdaptorBase::BrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.br", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr BrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::LoopAnnotationAttr BrOpGenericAdaptorBase::getLoopAnnotationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BrOp::getLoopAnnotationAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::LoopAnnotationAttr>();
  return attr;
}

::std::optional<::mlir::LLVM::LoopAnnotationAttr> BrOpGenericAdaptorBase::getLoopAnnotation() {
  auto attr = getLoopAnnotationAttr();
  return attr ? ::std::optional<::mlir::LLVM::LoopAnnotationAttr>(attr.cast<::mlir::LLVM::LoopAnnotationAttr>()) : (::std::nullopt);
}

} // namespace detail
BrOpAdaptor::BrOpAdaptor(BrOp op) : BrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BrOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_loop_annotation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BrOp::getLoopAnnotationAttrName(*odsOpName)) {
      tblgen_loop_annotation = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_loop_annotation && !((tblgen_loop_annotation.isa<::mlir::LLVM::LoopAnnotationAttr>())))
    return emitError(loc, "'llvm.br' op ""attribute 'loop_annotation' failed to satisfy constraint: ");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BrOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range BrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BrOp::getDestOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange BrOp::getDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BrOp::getDest() {
  return (*this)->getSuccessor(0);
}

::mlir::LLVM::LoopAnnotationAttr BrOp::getLoopAnnotationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAnnotationAttrName()).dyn_cast_or_null<::mlir::LLVM::LoopAnnotationAttr>();
}

::std::optional<::mlir::LLVM::LoopAnnotationAttr> BrOp::getLoopAnnotation() {
  auto attr = getLoopAnnotationAttr();
  return attr ? ::std::optional<::mlir::LLVM::LoopAnnotationAttr>(attr.cast<::mlir::LLVM::LoopAnnotationAttr>()) : (::std::nullopt);
}

void BrOp::setLoopAnnotationAttr(::mlir::LLVM::LoopAnnotationAttr attr) {
  (*this)->setAttr(getLoopAnnotationAttrName(), attr);
}

::mlir::Attribute BrOp::removeLoopAnnotationAttr() {
  return (*this)->removeAttr(getLoopAnnotationAttrName());
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Block *dest) {
      build(odsBuilder, odsState, ValueRange(), dest);
    
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, Block *dest) {
      build(odsBuilder, odsState, operands, /*loop_annotation=*/{}, dest);
    
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange destOperands, /*optional*/::mlir::LLVM::LoopAnnotationAttr loop_annotation, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  if (loop_annotation) {
    odsState.addAttribute(getLoopAnnotationAttrName(odsState.name), loop_annotation);
  }
  odsState.addSuccessors(dest);
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange destOperands, /*optional*/::mlir::LLVM::LoopAnnotationAttr loop_annotation, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  if (loop_annotation) {
    odsState.addAttribute(getLoopAnnotationAttrName(odsState.name), loop_annotation);
  }
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BrOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_loop_annotation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getLoopAnnotationAttrName()) {
      tblgen_loop_annotation = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps8(*this, tblgen_loop_annotation, "loop_annotation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult BrOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> destOperandsOperands;
  ::llvm::SMLoc destOperandsOperandsLoc;
  (void)destOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> destOperandsTypes;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  destOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(destOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(destOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  if (!getDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getDestOperands().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void BrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::BrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CallOpGenericAdaptorBase::CallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.call", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CallOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr CallOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr CallOpGenericAdaptorBase::getCalleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CallOp::getCalleeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > CallOpGenericAdaptorBase::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::LLVM::FastmathFlagsAttr CallOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CallOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags CallOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

::mlir::ElementsAttr CallOpGenericAdaptorBase::getBranchWeightsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CallOp::getBranchWeightsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::std::optional< ::mlir::ElementsAttr > CallOpGenericAdaptorBase::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

} // namespace detail
CallOpAdaptor::CallOpAdaptor(CallOp op) : CallOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CallOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_weights;
  ::mlir::Attribute tblgen_callee;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CallOp::getBranchWeightsAttrName(*odsOpName)) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CallOp::getCalleeAttrName(*odsOpName)) {
      tblgen_callee = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CallOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_callee && !((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.call' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");

  if (tblgen_branch_weights && !((tblgen_branch_weights.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.call' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CallOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CallOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CallOp::getResult() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : ::llvm::cast<::mlir::Value>(*results.begin());
}

::mlir::FlatSymbolRefAttr CallOp::getCalleeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCalleeAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::std::optional< ::llvm::StringRef > CallOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::LLVM::FastmathFlagsAttr CallOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags CallOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

::mlir::ElementsAttr CallOp::getBranchWeightsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchWeightsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::std::optional< ::mlir::ElementsAttr > CallOp::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

void CallOp::setCalleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

void CallOp::setCallee(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCalleeAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCalleeAttrName());
}

void CallOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void CallOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void CallOp::setBranchWeightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getBranchWeightsAttrName(), attr);
}

::mlir::Attribute CallOp::removeCalleeAttr() {
  return (*this)->removeAttr(getCalleeAttrName());
}

::mlir::Attribute CallOp::removeBranchWeightsAttr() {
  return (*this)->removeAttr(getBranchWeightsAttrName());
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags, /*optional*/::mlir::ElementsAttr branch_weights) {
  odsState.addOperands(odsArg_0);
  if (callee) {
    odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  if (result)
    odsState.addTypes(result);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags, /*optional*/::mlir::ElementsAttr branch_weights) {
  odsState.addOperands(odsArg_0);
  if (callee) {
    odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addTypes(resultTypes);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FastmathFlags fastmathFlags, /*optional*/::mlir::ElementsAttr branch_weights) {
  odsState.addOperands(odsArg_0);
  if (callee) {
    odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  if (result)
    odsState.addTypes(result);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FastmathFlags fastmathFlags, /*optional*/::mlir::ElementsAttr branch_weights) {
  odsState.addOperands(odsArg_0);
  if (callee) {
    odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addTypes(resultTypes);
}

void CallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

void CallOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[2])) {
    attributes.append(attrNames[2], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult CallOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_weights;
  ::mlir::Attribute tblgen_callee;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBranchWeightsAttrName()) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCalleeAttrName()) {
      tblgen_callee = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_callee, "callee")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps10(*this, tblgen_branch_weights, "branch_weights")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CallOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CondBrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CondBrOpGenericAdaptorBase::CondBrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.cond_br", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CondBrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CondBrOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CondBrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr CondBrOpGenericAdaptorBase::getBranchWeightsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CondBrOp::getBranchWeightsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::std::optional< ::mlir::ElementsAttr > CondBrOpGenericAdaptorBase::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

::mlir::LLVM::LoopAnnotationAttr CondBrOpGenericAdaptorBase::getLoopAnnotationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CondBrOp::getLoopAnnotationAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::LoopAnnotationAttr>();
  return attr;
}

::std::optional<::mlir::LLVM::LoopAnnotationAttr> CondBrOpGenericAdaptorBase::getLoopAnnotation() {
  auto attr = getLoopAnnotationAttr();
  return attr ? ::std::optional<::mlir::LLVM::LoopAnnotationAttr>(attr.cast<::mlir::LLVM::LoopAnnotationAttr>()) : (::std::nullopt);
}

} // namespace detail
CondBrOpAdaptor::CondBrOpAdaptor(CondBrOp op) : CondBrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CondBrOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_branch_weights;
  ::mlir::Attribute tblgen_loop_annotation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.cond_br' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CondBrOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CondBrOp::getBranchWeightsAttrName(*odsOpName)) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CondBrOp::getLoopAnnotationAttrName(*odsOpName)) {
      tblgen_loop_annotation = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'llvm.cond_br' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_branch_weights && !((tblgen_branch_weights.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.cond_br' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");

  if (tblgen_loop_annotation && !((tblgen_loop_annotation.isa<::mlir::LLVM::LoopAnnotationAttr>())))
    return emitError(loc, "'llvm.cond_br' op ""attribute 'loop_annotation' failed to satisfy constraint: ");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CondBrOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CondBrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> CondBrOp::getCondition() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range CondBrOp::getTrueDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CondBrOp::getFalseDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CondBrOp::getConditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CondBrOp::getTrueDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CondBrOp::getFalseDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CondBrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CondBrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CondBrOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CondBrOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::ElementsAttr CondBrOp::getBranchWeightsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBranchWeightsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::std::optional< ::mlir::ElementsAttr > CondBrOp::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

::mlir::LLVM::LoopAnnotationAttr CondBrOp::getLoopAnnotationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getLoopAnnotationAttrName()).dyn_cast_or_null<::mlir::LLVM::LoopAnnotationAttr>();
}

::std::optional<::mlir::LLVM::LoopAnnotationAttr> CondBrOp::getLoopAnnotation() {
  auto attr = getLoopAnnotationAttr();
  return attr ? ::std::optional<::mlir::LLVM::LoopAnnotationAttr>(attr.cast<::mlir::LLVM::LoopAnnotationAttr>()) : (::std::nullopt);
}

void CondBrOp::setBranchWeightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getBranchWeightsAttrName(), attr);
}

void CondBrOp::setLoopAnnotationAttr(::mlir::LLVM::LoopAnnotationAttr attr) {
  (*this)->setAttr(getLoopAnnotationAttrName(), attr);
}

::mlir::Attribute CondBrOp::removeBranchWeightsAttr() {
  return (*this)->removeAttr(getBranchWeightsAttrName());
}

::mlir::Attribute CondBrOp::removeLoopAnnotationAttr() {
  return (*this)->removeAttr(getLoopAnnotationAttrName());
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, ValueRange trueOperands, ValueRange falseOperands, ElementsAttr branchWeights, Block *trueDest, Block *falseDest) {
      build(odsBuilder, odsState, condition, trueOperands, falseOperands, branchWeights,
      {}, trueDest, falseDest);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, /*optional*/::mlir::LLVM::LoopAnnotationAttr loop_annotation, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  if (loop_annotation) {
    odsState.addAttribute(getLoopAnnotationAttrName(odsState.name), loop_annotation);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, /*optional*/::mlir::LLVM::LoopAnnotationAttr loop_annotation, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  if (loop_annotation) {
    odsState.addAttribute(getLoopAnnotationAttrName(odsState.name), loop_annotation);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CondBrOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_branch_weights;
  ::mlir::Attribute tblgen_loop_annotation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBranchWeightsAttrName()) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getLoopAnnotationAttrName()) {
      tblgen_loop_annotation = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps10(*this, tblgen_branch_weights, "branch_weights")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps8(*this, tblgen_loop_annotation, "loop_annotation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CondBrOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CondBrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand conditionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> conditionOperands(conditionRawOperands);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::ElementsAttr branch_weightsAttr;
  ::mlir::Block *trueDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> trueDestOperandsOperands;
  ::llvm::SMLoc trueDestOperandsOperandsLoc;
  (void)trueDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> trueDestOperandsTypes;
  ::mlir::Block *falseDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> falseDestOperandsOperands;
  ::llvm::SMLoc falseDestOperandsOperandsLoc;
  (void)falseDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> falseDestOperandsTypes;

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("weights"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(branch_weightsAttr, ::mlir::Type{}, "branch_weights",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(trueDestSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(trueDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(falseDestSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(falseDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, conditionOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CondBrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCondition();
  if ((*this)->getAttr("branch_weights")) {
    _odsPrinter << ' ' << "weights";
    _odsPrinter << "(";
  _odsPrinter.printStrippedAttrOrType(getBranchWeightsAttr());
    _odsPrinter << ")";
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTrueDest();
  if (!getTrueDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getTrueDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getTrueDestOperands().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFalseDest();
  if (!getFalseDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getFalseDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getFalseDestOperands().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("branch_weights");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CondBrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CondBrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConstantOpGenericAdaptorBase::ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.constant", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ConstantOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ConstantOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute ConstantOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantOp::getValueAttrName(*odsOpName)).cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute ConstantOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.constant' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((true)))
    return emitError(loc, "'llvm.mlir.constant' op ""attribute 'value' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Attribute ConstantOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, int64_t value) {
      build(odsBuilder, odsState, type, odsBuilder.getIntegerAttr(type, value));
    
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, const APInt &value) {
      build(odsBuilder, odsState, type, odsBuilder.getIntegerAttr(type, value));
    
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, const APFloat &value) {
      build(odsBuilder, odsState, type, odsBuilder.getFloatAttr(type, value));
    
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypedAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(res);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps11(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}, "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void ConstantOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter.printAttribute(getValueAttr());
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ConstantOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExtractElementOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractElementOpGenericAdaptorBase::ExtractElementOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.extractelement", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractElementOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExtractElementOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ExtractElementOpAdaptor::ExtractElementOpAdaptor(ExtractElementOp op) : ExtractElementOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::getVector() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IntegerType> ExtractElementOp::getPosition() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ExtractElementOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractElementOp::getPositionMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);
  odsState.addTypes(res);
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ExtractElementOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ExtractElementOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ExtractElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(LLVM::getVectorElementType((*this->getODSOperands(0).begin()).getType()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches vector element type");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractElementOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ExtractElementOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = LLVM::getVectorElementType(operands[0].getType());
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ExtractElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand positionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> positionOperands(positionRawOperands);  ::llvm::SMLoc positionOperandsLoc;
  (void)positionOperandsLoc;
  ::mlir::Type positionRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> positionTypes(positionRawTypes);
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  positionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(positionRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    positionRawTypes[0] = type;
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  for (::mlir::Type type : vectorTypes) {
    (void)type;
    if (!((::mlir::LLVM::isCompatibleVectorType(type)))) {
      return parser.emitError(parser.getNameLoc()) << "'vector' must be LLVM dialect-compatible vector type, but got " << type;
    }
  }
  result.addTypes(LLVM::getVectorElementType(vectorTypes[0]));
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(positionOperands, positionTypes, positionOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractElementOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << "[";
  _odsPrinter << getPosition();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getPosition().getType();
    if (auto validType = type.dyn_cast<::mlir::IntegerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExtractElementOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ExtractElementOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExtractValueOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractValueOpGenericAdaptorBase::ExtractValueOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.extractvalue", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractValueOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExtractValueOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ExtractValueOpGenericAdaptorBase::getPositionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExtractValueOp::getPositionAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ExtractValueOpGenericAdaptorBase::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

} // namespace detail
ExtractValueOpAdaptor::ExtractValueOpAdaptor(ExtractValueOp op) : ExtractValueOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractValueOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.extractvalue' op ""requires attribute 'position'");
    if (namedAttrIt->getName() == ExtractValueOp::getPositionAttrName(*odsOpName)) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_position && !((tblgen_position.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'llvm.extractvalue' op ""attribute 'position' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractValueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractValueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOp::getContainer() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExtractValueOp::getContainerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractValueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractValueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr ExtractValueOp::getPositionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPositionAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ExtractValueOp::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

void ExtractValueOp::setPositionAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPositionAttrName(), attr);
}

void ExtractValueOp::setPosition(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPositionAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::DenseI64ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  odsState.addTypes(res);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::DenseI64ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::llvm::ArrayRef<int64_t> position) {
  odsState.addOperands(container);
  odsState.addAttribute(getPositionAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(position));
  odsState.addTypes(res);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::llvm::ArrayRef<int64_t> position) {
  odsState.addOperands(container);
  odsState.addAttribute(getPositionAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(position));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractValueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractValueOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'position'");
    if (namedAttrIt->getName() == getPositionAttrName()) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps12(*this, tblgen_position, "position")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractValueOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExtractValueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand containerRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> containerOperands(containerRawOperands);  ::llvm::SMLoc containerOperandsLoc;
  (void)containerOperandsLoc;
  ::mlir::DenseI64ArrayAttr positionAttr;
  ::mlir::Type containerRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> containerTypes(containerRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  containerOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(containerRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(positionAttr, ::mlir::Type{}, "position",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    containerRawTypes[0] = type;
  }
  {
    if (parseInsertExtractValueElementType(parser, resRawTypes[0], containerRawTypes[0], positionAttr))
      return ::mlir::failure();
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(containerOperands, containerTypes, containerOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractValueOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getContainer();
_odsPrinter.printStrippedAttrOrType(getPositionAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("position");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getContainer().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  printInsertExtractValueElementType(_odsPrinter, *this, getRes().getType(), getContainer().getType(), getPositionAttr());
}

void ExtractValueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ExtractValueOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FAddOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FAddOpGenericAdaptorBase::FAddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fadd", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FAddOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FAddOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr FAddOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FAddOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FAddOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FAddOpAdaptor::FAddOpAdaptor(FAddOp op) : FAddOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FAddOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FAddOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.fadd' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FAddOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FAddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FAddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr FAddOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FAddOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FAddOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FAddOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FAddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FAddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FAddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FAddOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FAddOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FAddOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FAddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FAddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FCmpOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FCmpOpGenericAdaptorBase::FCmpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fcmp", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FCmpOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FCmpOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FCmpPredicateAttr FCmpOpGenericAdaptorBase::getPredicateAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FCmpOp::getPredicateAttrName(*odsOpName)).cast<::mlir::LLVM::FCmpPredicateAttr>();
  return attr;
}

::mlir::LLVM::FCmpPredicate FCmpOpGenericAdaptorBase::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

::mlir::LLVM::FastmathFlagsAttr FCmpOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, FCmpOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FCmpOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FCmpOpAdaptor::FCmpOpAdaptor(FCmpOp op) : FCmpOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FCmpOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.fcmp' op ""requires attribute 'predicate'");
    if (namedAttrIt->getName() == FCmpOp::getPredicateAttrName(*odsOpName)) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == FCmpOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_predicate && !((tblgen_predicate.isa<::mlir::LLVM::FCmpPredicateAttr>())))
    return emitError(loc, "'llvm.fcmp' op ""attribute 'predicate' failed to satisfy constraint: llvm.fcmp comparison predicate");

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.fcmp' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FCmpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FCmpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FCmpOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FCmpOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FCmpOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FCmpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FCmpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FCmpPredicateAttr FCmpOp::getPredicateAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPredicateAttrName()).cast<::mlir::LLVM::FCmpPredicateAttr>();
}

::mlir::LLVM::FCmpPredicate FCmpOp::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

::mlir::LLVM::FastmathFlagsAttr FCmpOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FCmpOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FCmpOp::setPredicateAttr(::mlir::LLVM::FCmpPredicateAttr attr) {
  (*this)->setAttr(getPredicateAttrName(), attr);
}

void FCmpOp::setPredicate(::mlir::LLVM::FCmpPredicate attrValue) {
  (*this)->setAttr(getPredicateAttrName(), ::mlir::LLVM::FCmpPredicateAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FCmpOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FCmpOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FCmpOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FCmpOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FCmpOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FCmpOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FCmpOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'predicate'");
    if (namedAttrIt->getName() == getPredicateAttrName()) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps13(*this, tblgen_predicate, "predicate")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(::getI1SameShape((*this->getODSOperands(0).begin()).getType()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::LogicalResult FCmpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FCmpOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = ::getI1SameShape(operands[0].getType());
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void FCmpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FCmpOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FDivOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FDivOpGenericAdaptorBase::FDivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fdiv", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FDivOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FDivOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr FDivOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FDivOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FDivOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FDivOpAdaptor::FDivOpAdaptor(FDivOp op) : FDivOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FDivOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FDivOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.fdiv' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FDivOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr FDivOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FDivOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FDivOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FDivOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FDivOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FDivOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FDivOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FMulOpGenericAdaptorBase::FMulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FMulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FMulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr FMulOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FMulOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FMulOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FMulOpAdaptor::FMulOpAdaptor(FMulOp op) : FMulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FMulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FMulOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.fmul' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FMulOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FMulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr FMulOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FMulOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FMulOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FMulOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FMulOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FMulOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FMulOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FMulOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FMulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FMulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FMulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FNegOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FNegOpGenericAdaptorBase::FNegOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fneg", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FNegOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FNegOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr FNegOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FNegOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FNegOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FNegOpAdaptor::FNegOpAdaptor(FNegOp op) : FNegOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FNegOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FNegOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.fneg' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FNegOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FNegOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOp::getOperand() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange FNegOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FNegOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FNegOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr FNegOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FNegOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FNegOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FNegOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value operand, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(operand);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(operand);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FNegOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(operand);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FNegOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FNegOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FNegOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FNegOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FNegOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FNegOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FNegOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(operandOperands, resTypes[0], operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FNegOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FNegOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FNegOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPExtOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FPExtOpGenericAdaptorBase::FPExtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fpext", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FPExtOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FPExtOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FPExtOpAdaptor::FPExtOpAdaptor(FPExtOp op) : FPExtOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FPExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange FPExtOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPExtOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPToSIOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FPToSIOpGenericAdaptorBase::FPToSIOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fptosi", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FPToSIOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FPToSIOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FPToSIOpAdaptor::FPToSIOpAdaptor(FPToSIOp op) : FPToSIOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FPToSIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPToSIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToSIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange FPToSIOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPToSIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToSIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToSIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPToSIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPToSIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPToSIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPToSIOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPToSIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPToSIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPToUIOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FPToUIOpGenericAdaptorBase::FPToUIOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fptoui", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FPToUIOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FPToUIOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FPToUIOpAdaptor::FPToUIOpAdaptor(FPToUIOp op) : FPToUIOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FPToUIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPToUIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToUIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange FPToUIOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPToUIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToUIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToUIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPToUIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPToUIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPToUIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPToUIOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPToUIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPToUIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPTruncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FPTruncOpGenericAdaptorBase::FPTruncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fptrunc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FPTruncOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FPTruncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FPTruncOpAdaptor::FPTruncOpAdaptor(FPTruncOp op) : FPTruncOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FPTruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPTruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPTruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange FPTruncOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPTruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPTruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPTruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPTruncOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPTruncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPTruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPTruncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPTruncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPTruncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FRemOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FRemOpGenericAdaptorBase::FRemOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.frem", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FRemOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FRemOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr FRemOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FRemOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FRemOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FRemOpAdaptor::FRemOpAdaptor(FRemOp op) : FRemOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FRemOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FRemOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.frem' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FRemOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr FRemOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FRemOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FRemOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FRemOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FRemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FRemOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FRemOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FRemOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FRemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FRemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FSubOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FSubOpGenericAdaptorBase::FSubOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fsub", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FSubOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FSubOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr FSubOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FSubOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags FSubOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
FSubOpAdaptor::FSubOpAdaptor(FSubOp op) : FSubOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FSubOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FSubOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.fsub' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FSubOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FSubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FSubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr FSubOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags FSubOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void FSubOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FSubOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FSubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FSubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FSubOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void FSubOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult FSubOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FSubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FSubOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FSubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FSubOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FSubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FSubOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FenceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FenceOpGenericAdaptorBase::FenceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fence", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FenceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FenceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicOrderingAttr FenceOpGenericAdaptorBase::getOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FenceOp::getOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrdering FenceOpGenericAdaptorBase::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr FenceOpGenericAdaptorBase::getSyncscopeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, FenceOp::getSyncscopeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > FenceOpGenericAdaptorBase::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
FenceOpAdaptor::FenceOpAdaptor(FenceOp op) : FenceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FenceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.fence' op ""requires attribute 'ordering'");
    if (namedAttrIt->getName() == FenceOp::getOrderingAttrName(*odsOpName)) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FenceOp::getSyncscopeAttrName(*odsOpName)) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_ordering && !((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.fence' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_syncscope && !((tblgen_syncscope.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.fence' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FenceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FenceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FenceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FenceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LLVM::AtomicOrderingAttr FenceOp::getOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering FenceOp::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr FenceOp::getSyncscopeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSyncscopeAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > FenceOp::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void FenceOp::setOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getOrderingAttrName(), attr);
}

void FenceOp::setOrdering(::mlir::LLVM::AtomicOrdering attrValue) {
  (*this)->setAttr(getOrderingAttrName(), ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void FenceOp::setSyncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSyncscopeAttrName(), attr);
}

void FenceOp::setSyncscope(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSyncscopeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSyncscopeAttrName());
}

::mlir::Attribute FenceOp::removeSyncscopeAttr() {
  return (*this)->removeAttr(getSyncscopeAttrName());
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FenceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ordering'");
    if (namedAttrIt->getName() == getOrderingAttrName()) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSyncscopeAttrName()) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_ordering, "ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_syncscope, "syncscope")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult FenceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult FenceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr syncscopeAttr;
  ::mlir::LLVM::AtomicOrderingAttr orderingAttr;
  if (::mlir::succeeded(parser.parseOptionalKeyword("syncscope"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(syncscopeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "syncscope",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"not_atomic","unordered","monotonic","acquire","release","acq_rel","seq_cst"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "ordering", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'ordering' [not_atomic, unordered, monotonic, acquire, release, acq_rel, seq_cst]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicOrdering(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "ordering attribute specification: \"" << attrStr << '"';;

      orderingAttr = ::mlir::LLVM::AtomicOrderingAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("ordering", orderingAttr);
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void FenceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("syncscope")) {
    _odsPrinter << ' ' << "syncscope";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getSyncscopeAttr());
    _odsPrinter << ")";
  }
  _odsPrinter << ' ';

  {
    auto caseValue = getOrdering();
    auto caseValueStr = stringifyAtomicOrdering(caseValue);
    _odsPrinter << caseValueStr;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("syncscope");
  elidedAttrs.push_back("ordering");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FenceOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FreezeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FreezeOpGenericAdaptorBase::FreezeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.freeze", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FreezeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FreezeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FreezeOpAdaptor::FreezeOpAdaptor(FreezeOp op) : FreezeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FreezeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FreezeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FreezeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOp::getVal() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange FreezeOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FreezeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FreezeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val) {
  odsState.addOperands(val);
  odsState.addTypes(res);
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value val) {
  odsState.addOperands(val);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FreezeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val) {
  odsState.addOperands(val);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FreezeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FreezeOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FreezeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FreezeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FreezeOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FreezeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valOperands(valRawOperands);  ::llvm::SMLoc valOperandsLoc;
  (void)valOperandsLoc;
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);

  valOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valRawTypes[0] = type;
  }
  result.addTypes(valTypes[0]);
  if (parser.resolveOperands(valOperands, valTypes, valOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FreezeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVal();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVal().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FreezeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GEPOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GEPOpGenericAdaptorBase::GEPOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.getelementptr", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GEPOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr GEPOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI32ArrayAttr GEPOpGenericAdaptorBase::getRawConstantIndicesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GEPOp::getRawConstantIndicesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int32_t> GEPOpGenericAdaptorBase::getRawConstantIndices() {
  auto attr = getRawConstantIndicesAttr();
  return attr;
}

::mlir::TypeAttr GEPOpGenericAdaptorBase::getElemTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GEPOp::getElemTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::std::optional<::mlir::Type> GEPOpGenericAdaptorBase::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::std::optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::std::nullopt);
}

::mlir::UnitAttr GEPOpGenericAdaptorBase::getInboundsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GEPOp::getInboundsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GEPOpGenericAdaptorBase::getInbounds() {
  auto attr = getInboundsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
GEPOpAdaptor::GEPOpAdaptor(GEPOp op) : GEPOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GEPOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_rawConstantIndices;
  ::mlir::Attribute tblgen_elem_type;
  ::mlir::Attribute tblgen_inbounds;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.getelementptr' op ""requires attribute 'rawConstantIndices'");
    if (namedAttrIt->getName() == GEPOp::getRawConstantIndicesAttrName(*odsOpName)) {
      tblgen_rawConstantIndices = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GEPOp::getElemTypeAttrName(*odsOpName)) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GEPOp::getInboundsAttrName(*odsOpName)) {
      tblgen_inbounds = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_rawConstantIndices && !((tblgen_rawConstantIndices.isa<::mlir::DenseI32ArrayAttr>())))
    return emitError(loc, "'llvm.getelementptr' op ""attribute 'rawConstantIndices' failed to satisfy constraint: i32 dense array attribute");

  if (tblgen_elem_type && !(((tblgen_elem_type.isa<::mlir::TypeAttr>())) && ((tblgen_elem_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'llvm.getelementptr' op ""attribute 'elem_type' failed to satisfy constraint: any type attribute");

  if (tblgen_inbounds && !((tblgen_inbounds.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.getelementptr' op ""attribute 'inbounds' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GEPOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range GEPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOp::getBase() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range GEPOp::getDynamicIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GEPOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GEPOp::getDynamicIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GEPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GEPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::DenseI32ArrayAttr GEPOp::getRawConstantIndicesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getRawConstantIndicesAttrName()).cast<::mlir::DenseI32ArrayAttr>();
}

::llvm::ArrayRef<int32_t> GEPOp::getRawConstantIndices() {
  auto attr = getRawConstantIndicesAttr();
  return attr;
}

::mlir::TypeAttr GEPOp::getElemTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getElemTypeAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::std::optional<::mlir::Type> GEPOp::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::std::optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::std::nullopt);
}

::mlir::UnitAttr GEPOp::getInboundsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInboundsAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GEPOp::getInbounds() {
  auto attr = getInboundsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void GEPOp::setRawConstantIndicesAttr(::mlir::DenseI32ArrayAttr attr) {
  (*this)->setAttr(getRawConstantIndicesAttrName(), attr);
}

void GEPOp::setRawConstantIndices(::llvm::ArrayRef<int32_t> attrValue) {
  (*this)->setAttr(getRawConstantIndicesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue));
}

void GEPOp::setElemTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getElemTypeAttrName(), attr);
}

void GEPOp::setElemType(::std::optional<::mlir::Type> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getElemTypeAttrName(), ::mlir::TypeAttr::get(*attrValue));
    (*this)->removeAttr(getElemTypeAttrName());
}

void GEPOp::setInboundsAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInboundsAttrName(), attr);
}

void GEPOp::setInbounds(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInboundsAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInboundsAttrName());
}

::mlir::Attribute GEPOp::removeElemTypeAttr() {
  return (*this)->removeAttr(getElemTypeAttrName());
}

::mlir::Attribute GEPOp::removeInboundsAttr() {
  return (*this)->removeAttr(getInboundsAttrName());
}

::mlir::LogicalResult GEPOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_rawConstantIndices;
  ::mlir::Attribute tblgen_elem_type;
  ::mlir::Attribute tblgen_inbounds;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'rawConstantIndices'");
    if (namedAttrIt->getName() == getRawConstantIndicesAttrName()) {
      tblgen_rawConstantIndices = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getElemTypeAttrName()) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getInboundsAttrName()) {
      tblgen_inbounds = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_rawConstantIndices, "rawConstantIndices")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps3(*this, tblgen_elem_type, "elem_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_inbounds, "inbounds")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GEPOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GEPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dynamicIndicesOperands;
  ::llvm::SMLoc dynamicIndicesOperandsLoc;
  (void)dynamicIndicesOperandsLoc;
  ::mlir::DenseI32ArrayAttr rawConstantIndicesAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  ::mlir::TypeAttr elem_typeAttr;
  if (::mlir::succeeded(parser.parseOptionalKeyword("inbounds"))) {
    result.addAttribute("inbounds", parser.getBuilder().getUnitAttr());
  }

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    dynamicIndicesOperandsLoc = parser.getCurrentLocation();
    if (parseGEPIndices(parser, dynamicIndicesOperands, rawConstantIndicesAttr))
      return ::mlir::failure();
    result.addAttribute("rawConstantIndices", rawConstantIndicesAttr);
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseCustomAttributeWithFallback(elem_typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "elem_type",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(baseOperands, dynamicIndicesOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GEPOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("inbounds")) {
    _odsPrinter << ' ' << "inbounds";
  }
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  printGEPIndices(_odsPrinter, *this, getDynamicIndices(), getRawConstantIndicesAttr());
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("inbounds");
  elidedAttrs.push_back("rawConstantIndices");
  elidedAttrs.push_back("elem_type");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInboundsAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inbounds");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
  if ((*this)->getAttr("elem_type")) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getElemTypeAttr());
  }
}

void GEPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GEPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalCtorsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GlobalCtorsOpGenericAdaptorBase::GlobalCtorsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.global_ctors", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GlobalCtorsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GlobalCtorsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GlobalCtorsOpGenericAdaptorBase::getCtorsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GlobalCtorsOp::getCtorsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GlobalCtorsOpGenericAdaptorBase::getCtors() {
  auto attr = getCtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalCtorsOpGenericAdaptorBase::getPrioritiesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GlobalCtorsOp::getPrioritiesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GlobalCtorsOpGenericAdaptorBase::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

} // namespace detail
GlobalCtorsOpAdaptor::GlobalCtorsOpAdaptor(GlobalCtorsOp op) : GlobalCtorsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GlobalCtorsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ctors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_ctors' op ""requires attribute 'ctors'");
    if (namedAttrIt->getName() == GlobalCtorsOp::getCtorsAttrName(*odsOpName)) {
      tblgen_ctors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_ctors' op ""requires attribute 'priorities'");
    if (namedAttrIt->getName() == GlobalCtorsOp::getPrioritiesAttrName(*odsOpName)) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_ctors && !(((tblgen_ctors.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ctors.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.mlir.global_ctors' op ""attribute 'ctors' failed to satisfy constraint: flat symbol ref array attribute");

  if (tblgen_priorities && !(((tblgen_priorities.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_priorities.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'llvm.mlir.global_ctors' op ""attribute 'priorities' failed to satisfy constraint: 32-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalCtorsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalCtorsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalCtorsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalCtorsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr GlobalCtorsOp::getCtorsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCtorsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalCtorsOp::getCtors() {
  auto attr = getCtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalCtorsOp::getPrioritiesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getPrioritiesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalCtorsOp::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

void GlobalCtorsOp::setCtorsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCtorsAttrName(), attr);
}

void GlobalCtorsOp::setPrioritiesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPrioritiesAttrName(), attr);
}

void GlobalCtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ArrayAttr ctors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getCtorsAttrName(odsState.name), ctors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
}

void GlobalCtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr ctors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getCtorsAttrName(odsState.name), ctors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalCtorsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalCtorsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ctors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ctors'");
    if (namedAttrIt->getName() == getCtorsAttrName()) {
      tblgen_ctors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'priorities'");
    if (namedAttrIt->getName() == getPrioritiesAttrName()) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps15(*this, tblgen_ctors, "ctors")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps16(*this, tblgen_priorities, "priorities")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult GlobalCtorsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GlobalCtorsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalCtorsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GlobalCtorsOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalDtorsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GlobalDtorsOpGenericAdaptorBase::GlobalDtorsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.global_dtors", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GlobalDtorsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GlobalDtorsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GlobalDtorsOpGenericAdaptorBase::getDtorsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GlobalDtorsOp::getDtorsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GlobalDtorsOpGenericAdaptorBase::getDtors() {
  auto attr = getDtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalDtorsOpGenericAdaptorBase::getPrioritiesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GlobalDtorsOp::getPrioritiesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GlobalDtorsOpGenericAdaptorBase::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

} // namespace detail
GlobalDtorsOpAdaptor::GlobalDtorsOpAdaptor(GlobalDtorsOp op) : GlobalDtorsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GlobalDtorsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dtors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_dtors' op ""requires attribute 'dtors'");
    if (namedAttrIt->getName() == GlobalDtorsOp::getDtorsAttrName(*odsOpName)) {
      tblgen_dtors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_dtors' op ""requires attribute 'priorities'");
    if (namedAttrIt->getName() == GlobalDtorsOp::getPrioritiesAttrName(*odsOpName)) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dtors && !(((tblgen_dtors.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dtors.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.mlir.global_dtors' op ""attribute 'dtors' failed to satisfy constraint: flat symbol ref array attribute");

  if (tblgen_priorities && !(((tblgen_priorities.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_priorities.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'llvm.mlir.global_dtors' op ""attribute 'priorities' failed to satisfy constraint: 32-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalDtorsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalDtorsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalDtorsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalDtorsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr GlobalDtorsOp::getDtorsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDtorsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalDtorsOp::getDtors() {
  auto attr = getDtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalDtorsOp::getPrioritiesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getPrioritiesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalDtorsOp::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

void GlobalDtorsOp::setDtorsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDtorsAttrName(), attr);
}

void GlobalDtorsOp::setPrioritiesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPrioritiesAttrName(), attr);
}

void GlobalDtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ArrayAttr dtors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getDtorsAttrName(odsState.name), dtors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
}

void GlobalDtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr dtors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getDtorsAttrName(odsState.name), dtors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalDtorsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalDtorsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dtors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dtors'");
    if (namedAttrIt->getName() == getDtorsAttrName()) {
      tblgen_dtors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'priorities'");
    if (namedAttrIt->getName() == getPrioritiesAttrName()) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps15(*this, tblgen_dtors, "dtors")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps16(*this, tblgen_priorities, "priorities")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult GlobalDtorsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GlobalDtorsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalDtorsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GlobalDtorsOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GlobalOpGenericAdaptorBase::GlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.global", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GlobalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GlobalOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr GlobalOpGenericAdaptorBase::getGlobalTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GlobalOp::getGlobalTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type GlobalOpGenericAdaptorBase::getGlobalType() {
  auto attr = getGlobalTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOpGenericAdaptorBase::getConstantAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getConstantAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GlobalOpGenericAdaptorBase::getConstant() {
  auto attr = getConstantAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::StringAttr GlobalOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, GlobalOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GlobalOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::LLVM::LinkageAttr GlobalOpGenericAdaptorBase::getLinkageAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GlobalOp::getLinkageAttrName(*odsOpName)).cast<::mlir::LLVM::LinkageAttr>();
  return attr;
}

::mlir::LLVM::Linkage GlobalOpGenericAdaptorBase::getLinkage() {
  auto attr = getLinkageAttr();
  return attr.getLinkage();
}

::mlir::UnitAttr GlobalOpGenericAdaptorBase::getDsoLocalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getDsoLocalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GlobalOpGenericAdaptorBase::getDsoLocal() {
  auto attr = getDsoLocalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr GlobalOpGenericAdaptorBase::getThreadLocal_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getThreadLocal_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GlobalOpGenericAdaptorBase::getThreadLocal_() {
  auto attr = getThreadLocal_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Attribute GlobalOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getValueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::Attribute>();
  return attr;
}

::std::optional<::mlir::Attribute> GlobalOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr ? ::std::optional<::mlir::Attribute>(attr) : (::std::nullopt);
}

::mlir::IntegerAttr GlobalOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> GlobalOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::IntegerAttr GlobalOpGenericAdaptorBase::getAddrSpaceAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getAddrSpaceAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GlobalOpGenericAdaptorBase::getAddrSpace() {
  auto attr = getAddrSpaceAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LLVM::UnnamedAddrAttr GlobalOpGenericAdaptorBase::getUnnamedAddrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getUnnamedAddrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::UnnamedAddrAttr>();
  return attr;
}

::std::optional<::mlir::LLVM::UnnamedAddr> GlobalOpGenericAdaptorBase::getUnnamedAddr() {
  auto attr = getUnnamedAddrAttr();
  return attr ? ::std::optional<::mlir::LLVM::UnnamedAddr>(attr.getValue()) : (::std::nullopt);
}

::mlir::StringAttr GlobalOpGenericAdaptorBase::getSectionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, GlobalOp::getSectionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > GlobalOpGenericAdaptorBase::getSection() {
  auto attr = getSectionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::LLVM::VisibilityAttr GlobalOpGenericAdaptorBase::getVisibility_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getVisibility_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::VisibilityAttr>();
  return attr;
}

::mlir::LLVM::Visibility GlobalOpGenericAdaptorBase::getVisibility_() {
  auto attr = getVisibility_Attr();
  return attr.getValue();
}

::mlir::Region &GlobalOpGenericAdaptorBase::getInitializer() {
  return *odsRegions[0];
}

::mlir::RegionRange GlobalOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
GlobalOpAdaptor::GlobalOpAdaptor(GlobalOp op) : GlobalOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GlobalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_type;
  ::mlir::Attribute tblgen_addr_space;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_constant;
  ::mlir::Attribute tblgen_dso_local;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'global_type'");
    if (namedAttrIt->getName() == GlobalOp::getGlobalTypeAttrName(*odsOpName)) {
      tblgen_global_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getAddrSpaceAttrName(*odsOpName)) {
      tblgen_addr_space = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getConstantAttrName(*odsOpName)) {
      tblgen_constant = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getDsoLocalAttrName(*odsOpName)) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_linkage;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'linkage'");
    if (namedAttrIt->getName() == GlobalOp::getLinkageAttrName(*odsOpName)) {
      tblgen_linkage = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_section;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == GlobalOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getSectionAttrName(*odsOpName)) {
      tblgen_section = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_thread_local_;
  ::mlir::Attribute tblgen_unnamed_addr;
  ::mlir::Attribute tblgen_value;
  ::mlir::Attribute tblgen_visibility_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getThreadLocal_AttrName(*odsOpName)) {
      tblgen_thread_local_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getUnnamedAddrAttrName(*odsOpName)) {
      tblgen_unnamed_addr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getVisibility_AttrName(*odsOpName)) {
      tblgen_visibility_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_global_type && !(((tblgen_global_type.isa<::mlir::TypeAttr>())) && ((tblgen_global_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'global_type' failed to satisfy constraint: any type attribute");

  if (tblgen_constant && !((tblgen_constant.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'constant' failed to satisfy constraint: unit attribute");

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_linkage && !((tblgen_linkage.isa<::mlir::LLVM::LinkageAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'linkage' failed to satisfy constraint: LLVM Linkage specification");

  if (tblgen_dso_local && !((tblgen_dso_local.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'dso_local' failed to satisfy constraint: unit attribute");

  if (tblgen_thread_local_ && !((tblgen_thread_local_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'thread_local_' failed to satisfy constraint: unit attribute");

  if (tblgen_value && !((true)))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'value' failed to satisfy constraint: any attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_addr_space && !((((tblgen_addr_space.isa<::mlir::IntegerAttr>())) && ((tblgen_addr_space.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_addr_space.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_unnamed_addr && !((tblgen_unnamed_addr.isa<::mlir::LLVM::UnnamedAddrAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'unnamed_addr' failed to satisfy constraint: LLVM GlobalValue UnnamedAddr");

  if (tblgen_section && !((tblgen_section.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'section' failed to satisfy constraint: string attribute");

  if (tblgen_visibility_ && !((tblgen_visibility_.isa<::mlir::LLVM::VisibilityAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'visibility_' failed to satisfy constraint: LLVM GlobalValue Visibility");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &GlobalOp::getInitializer() {
  return (*this)->getRegion(0);
}

::mlir::TypeAttr GlobalOp::getGlobalTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getGlobalTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type GlobalOp::getGlobalType() {
  auto attr = getGlobalTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOp::getConstantAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getConstantAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getConstant() {
  auto attr = getConstantAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::StringAttr GlobalOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GlobalOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::LLVM::LinkageAttr GlobalOp::getLinkageAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getLinkageAttrName()).cast<::mlir::LLVM::LinkageAttr>();
}

::mlir::LLVM::Linkage GlobalOp::getLinkage() {
  auto attr = getLinkageAttr();
  return attr.getLinkage();
}

::mlir::UnitAttr GlobalOp::getDsoLocalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getDsoLocalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getDsoLocal() {
  auto attr = getDsoLocalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr GlobalOp::getThreadLocal_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getThreadLocal_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getThreadLocal_() {
  auto attr = getThreadLocal_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Attribute GlobalOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getValueAttrName()).dyn_cast_or_null<::mlir::Attribute>();
}

::std::optional<::mlir::Attribute> GlobalOp::getValue() {
  auto attr = getValueAttr();
  return attr ? ::std::optional<::mlir::Attribute>(attr) : (::std::nullopt);
}

::mlir::IntegerAttr GlobalOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> GlobalOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::IntegerAttr GlobalOp::getAddrSpaceAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getAddrSpaceAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint32_t GlobalOp::getAddrSpace() {
  auto attr = getAddrSpaceAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LLVM::UnnamedAddrAttr GlobalOp::getUnnamedAddrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getUnnamedAddrAttrName()).dyn_cast_or_null<::mlir::LLVM::UnnamedAddrAttr>();
}

::std::optional<::mlir::LLVM::UnnamedAddr> GlobalOp::getUnnamedAddr() {
  auto attr = getUnnamedAddrAttr();
  return attr ? ::std::optional<::mlir::LLVM::UnnamedAddr>(attr.getValue()) : (::std::nullopt);
}

::mlir::StringAttr GlobalOp::getSectionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getSectionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > GlobalOp::getSection() {
  auto attr = getSectionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::LLVM::VisibilityAttr GlobalOp::getVisibility_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getVisibility_AttrName()).dyn_cast_or_null<::mlir::LLVM::VisibilityAttr>();
}

::mlir::LLVM::Visibility GlobalOp::getVisibility_() {
  auto attr = getVisibility_Attr();
  return attr.getValue();
}

void GlobalOp::setGlobalTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getGlobalTypeAttrName(), attr);
}

void GlobalOp::setGlobalType(::mlir::Type attrValue) {
  (*this)->setAttr(getGlobalTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void GlobalOp::setConstantAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getConstantAttrName(), attr);
}

void GlobalOp::setConstant(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getConstantAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getConstantAttrName());
}

void GlobalOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void GlobalOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void GlobalOp::setLinkageAttr(::mlir::LLVM::LinkageAttr attr) {
  (*this)->setAttr(getLinkageAttrName(), attr);
}

void GlobalOp::setLinkage(::mlir::LLVM::Linkage attrValue) {
  (*this)->setAttr(getLinkageAttrName(), ::mlir::LLVM::LinkageAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void GlobalOp::setDsoLocalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getDsoLocalAttrName(), attr);
}

void GlobalOp::setDsoLocal(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDsoLocalAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getDsoLocalAttrName());
}

void GlobalOp::setThreadLocal_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getThreadLocal_AttrName(), attr);
}

void GlobalOp::setThreadLocal_(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getThreadLocal_AttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getThreadLocal_AttrName());
}

void GlobalOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void GlobalOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void GlobalOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

void GlobalOp::setAddrSpaceAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAddrSpaceAttrName(), attr);
}

void GlobalOp::setAddrSpace(uint32_t attrValue) {
  (*this)->setAttr(getAddrSpaceAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void GlobalOp::setUnnamedAddrAttr(::mlir::LLVM::UnnamedAddrAttr attr) {
  (*this)->setAttr(getUnnamedAddrAttrName(), attr);
}

void GlobalOp::setUnnamedAddr(::std::optional<::mlir::LLVM::UnnamedAddr> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getUnnamedAddrAttrName(), ::mlir::LLVM::UnnamedAddrAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getUnnamedAddrAttrName());
}

void GlobalOp::setSectionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSectionAttrName(), attr);
}

void GlobalOp::setSection(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSectionAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSectionAttrName());
}

void GlobalOp::setVisibility_Attr(::mlir::LLVM::VisibilityAttr attr) {
  (*this)->setAttr(getVisibility_AttrName(), attr);
}

void GlobalOp::setVisibility_(::mlir::LLVM::Visibility attrValue) {
  (*this)->setAttr(getVisibility_AttrName(), ::mlir::LLVM::VisibilityAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

::mlir::Attribute GlobalOp::removeConstantAttr() {
  return (*this)->removeAttr(getConstantAttrName());
}

::mlir::Attribute GlobalOp::removeDsoLocalAttr() {
  return (*this)->removeAttr(getDsoLocalAttrName());
}

::mlir::Attribute GlobalOp::removeThreadLocal_Attr() {
  return (*this)->removeAttr(getThreadLocal_AttrName());
}

::mlir::Attribute GlobalOp::removeValueAttr() {
  return (*this)->removeAttr(getValueAttrName());
}

::mlir::Attribute GlobalOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute GlobalOp::removeUnnamedAddrAttr() {
  return (*this)->removeAttr(getUnnamedAddrAttrName());
}

::mlir::Attribute GlobalOp::removeSectionAttr() {
  return (*this)->removeAttr(getSectionAttrName());
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeAttr global_type, /*optional*/::mlir::UnitAttr constant, ::mlir::StringAttr sym_name, ::mlir::LLVM::LinkageAttr linkage, /*optional*/::mlir::UnitAttr dso_local, /*optional*/::mlir::UnitAttr thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, ::mlir::IntegerAttr addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section, ::mlir::LLVM::VisibilityAttr visibility_) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), global_type);
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkage);
  if (dso_local) {
    odsState.addAttribute(getDsoLocalAttrName(odsState.name), dso_local);
  }
  if (thread_local_) {
    odsState.addAttribute(getThreadLocal_AttrName(odsState.name), thread_local_);
  }
  if (value) {
    odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (addr_space) {
    odsState.addAttribute(getAddrSpaceAttrName(odsState.name), addr_space);
  }
  if (unnamed_addr) {
    odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
    odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  if (visibility_) {
    odsState.addAttribute(getVisibility_AttrName(odsState.name), visibility_);
  }
  (void)odsState.addRegion();
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr global_type, /*optional*/::mlir::UnitAttr constant, ::mlir::StringAttr sym_name, ::mlir::LLVM::LinkageAttr linkage, /*optional*/::mlir::UnitAttr dso_local, /*optional*/::mlir::UnitAttr thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, ::mlir::IntegerAttr addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section, ::mlir::LLVM::VisibilityAttr visibility_) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), global_type);
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkage);
  if (dso_local) {
    odsState.addAttribute(getDsoLocalAttrName(odsState.name), dso_local);
  }
  if (thread_local_) {
    odsState.addAttribute(getThreadLocal_AttrName(odsState.name), thread_local_);
  }
  if (value) {
    odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (addr_space) {
    odsState.addAttribute(getAddrSpaceAttrName(odsState.name), addr_space);
  }
  if (unnamed_addr) {
    odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
    odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  if (visibility_) {
    odsState.addAttribute(getVisibility_AttrName(odsState.name), visibility_);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type global_type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage, /*optional*/bool dso_local, /*optional*/bool thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, uint32_t addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section, ::mlir::LLVM::Visibility visibility_) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), ::mlir::TypeAttr::get(global_type));
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), ((constant) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getLinkageAttrName(odsState.name), ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), linkage));
  if (dso_local) {
    odsState.addAttribute(getDsoLocalAttrName(odsState.name), ((dso_local) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (thread_local_) {
    odsState.addAttribute(getThreadLocal_AttrName(odsState.name), ((thread_local_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (value) {
    odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addAttribute(getAddrSpaceAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), addr_space));
  if (unnamed_addr) {
    odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
    odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  odsState.addAttribute(getVisibility_AttrName(odsState.name), ::mlir::LLVM::VisibilityAttr::get(odsBuilder.getContext(), visibility_));
  (void)odsState.addRegion();
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type global_type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage, /*optional*/bool dso_local, /*optional*/bool thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, uint32_t addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section, ::mlir::LLVM::Visibility visibility_) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), ::mlir::TypeAttr::get(global_type));
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), ((constant) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getLinkageAttrName(odsState.name), ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), linkage));
  if (dso_local) {
    odsState.addAttribute(getDsoLocalAttrName(odsState.name), ((dso_local) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (thread_local_) {
    odsState.addAttribute(getThreadLocal_AttrName(odsState.name), ((thread_local_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (value) {
    odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addAttribute(getAddrSpaceAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), addr_space));
  if (unnamed_addr) {
    odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
    odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  odsState.addAttribute(getVisibility_AttrName(odsState.name), ::mlir::LLVM::VisibilityAttr::get(odsBuilder.getContext(), visibility_));
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void GlobalOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  }
  if (!attributes.get(attrNames[11])) {
    attributes.append(attrNames[11], ::mlir::LLVM::VisibilityAttr::get(odsBuilder.getContext(), mlir::LLVM::Visibility::Default));
  }
}

::mlir::LogicalResult GlobalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_type;
  ::mlir::Attribute tblgen_addr_space;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_constant;
  ::mlir::Attribute tblgen_dso_local;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global_type'");
    if (namedAttrIt->getName() == getGlobalTypeAttrName()) {
      tblgen_global_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAddrSpaceAttrName()) {
      tblgen_addr_space = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getConstantAttrName()) {
      tblgen_constant = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDsoLocalAttrName()) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_linkage;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'linkage'");
    if (namedAttrIt->getName() == getLinkageAttrName()) {
      tblgen_linkage = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_section;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getSectionAttrName()) {
      tblgen_section = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_thread_local_;
  ::mlir::Attribute tblgen_unnamed_addr;
  ::mlir::Attribute tblgen_value;
  ::mlir::Attribute tblgen_visibility_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getThreadLocal_AttrName()) {
      tblgen_thread_local_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUnnamedAddrAttrName()) {
      tblgen_unnamed_addr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVisibility_AttrName()) {
      tblgen_visibility_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps3(*this, tblgen_global_type, "global_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_constant, "constant")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps17(*this, tblgen_linkage, "linkage")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_dso_local, "dso_local")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_thread_local_, "thread_local_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps11(*this, tblgen_value, "value")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps18(*this, tblgen_addr_space, "addr_space")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps19(*this, tblgen_unnamed_addr, "unnamed_addr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_section, "section")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps20(*this, tblgen_visibility_, "visibility_")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LLVMOps0(*this, region, "initializer", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GlobalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GlobalOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ICmpOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ICmpOpGenericAdaptorBase::ICmpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.icmp", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ICmpOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ICmpOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::ICmpPredicateAttr ICmpOpGenericAdaptorBase::getPredicateAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ICmpOp::getPredicateAttrName(*odsOpName)).cast<::mlir::LLVM::ICmpPredicateAttr>();
  return attr;
}

::mlir::LLVM::ICmpPredicate ICmpOpGenericAdaptorBase::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

} // namespace detail
ICmpOpAdaptor::ICmpOpAdaptor(ICmpOp op) : ICmpOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ICmpOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.icmp' op ""requires attribute 'predicate'");
    if (namedAttrIt->getName() == ICmpOp::getPredicateAttrName(*odsOpName)) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_predicate && !((tblgen_predicate.isa<::mlir::LLVM::ICmpPredicateAttr>())))
    return emitError(loc, "'llvm.icmp' op ""attribute 'predicate' failed to satisfy constraint: lvm.icmp comparison predicate");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ICmpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ICmpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value ICmpOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ICmpOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ICmpOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ICmpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ICmpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::ICmpPredicateAttr ICmpOp::getPredicateAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPredicateAttrName()).cast<::mlir::LLVM::ICmpPredicateAttr>();
}

::mlir::LLVM::ICmpPredicate ICmpOp::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

void ICmpOp::setPredicateAttr(::mlir::LLVM::ICmpPredicateAttr attr) {
  (*this)->setAttr(getPredicateAttrName(), attr);
}

void ICmpOp::setPredicate(::mlir::LLVM::ICmpPredicate attrValue) {
  (*this)->setAttr(getPredicateAttrName(), ::mlir::LLVM::ICmpPredicateAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  odsState.addTypes(res);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ICmpOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addTypes(res);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ICmpOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ICmpOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ICmpOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'predicate'");
    if (namedAttrIt->getName() == getPredicateAttrName()) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps21(*this, tblgen_predicate, "predicate")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(::getI1SameShape((*this->getODSOperands(0).begin()).getType()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::LogicalResult ICmpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ICmpOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = ::getI1SameShape(operands[0].getType());
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void ICmpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ICmpOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InlineAsmOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InlineAsmOpGenericAdaptorBase::InlineAsmOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.inline_asm", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InlineAsmOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr InlineAsmOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr InlineAsmOpGenericAdaptorBase::getAsmStringAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, InlineAsmOp::getAsmStringAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef InlineAsmOpGenericAdaptorBase::getAsmString() {
  auto attr = getAsmStringAttr();
  return attr.getValue();
}

::mlir::StringAttr InlineAsmOpGenericAdaptorBase::getConstraintsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, InlineAsmOp::getConstraintsAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef InlineAsmOpGenericAdaptorBase::getConstraints() {
  auto attr = getConstraintsAttr();
  return attr.getValue();
}

::mlir::UnitAttr InlineAsmOpGenericAdaptorBase::getHasSideEffectsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, InlineAsmOp::getHasSideEffectsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool InlineAsmOpGenericAdaptorBase::getHasSideEffects() {
  auto attr = getHasSideEffectsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr InlineAsmOpGenericAdaptorBase::getIsAlignStackAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, InlineAsmOp::getIsAlignStackAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool InlineAsmOpGenericAdaptorBase::getIsAlignStack() {
  auto attr = getIsAlignStackAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::AsmDialectAttr InlineAsmOpGenericAdaptorBase::getAsmDialectAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, InlineAsmOp::getAsmDialectAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::AsmDialectAttr>();
  return attr;
}

::std::optional<::mlir::LLVM::AsmDialect> InlineAsmOpGenericAdaptorBase::getAsmDialect() {
  auto attr = getAsmDialectAttr();
  return attr ? ::std::optional<::mlir::LLVM::AsmDialect>(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr InlineAsmOpGenericAdaptorBase::getOperandAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, InlineAsmOp::getOperandAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > InlineAsmOpGenericAdaptorBase::getOperandAttrs() {
  auto attr = getOperandAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
InlineAsmOpAdaptor::InlineAsmOpAdaptor(InlineAsmOp op) : InlineAsmOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InlineAsmOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_asm_string;
  ::mlir::Attribute tblgen_asm_dialect;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.inline_asm' op ""requires attribute 'asm_string'");
    if (namedAttrIt->getName() == InlineAsmOp::getAsmStringAttrName(*odsOpName)) {
      tblgen_asm_string = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getAsmDialectAttrName(*odsOpName)) {
      tblgen_asm_dialect = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_constraints;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.inline_asm' op ""requires attribute 'constraints'");
    if (namedAttrIt->getName() == InlineAsmOp::getConstraintsAttrName(*odsOpName)) {
      tblgen_constraints = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_has_side_effects;
  ::mlir::Attribute tblgen_is_align_stack;
  ::mlir::Attribute tblgen_operand_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getHasSideEffectsAttrName(*odsOpName)) {
      tblgen_has_side_effects = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getIsAlignStackAttrName(*odsOpName)) {
      tblgen_is_align_stack = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getOperandAttrsAttrName(*odsOpName)) {
      tblgen_operand_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_asm_string && !((tblgen_asm_string.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'asm_string' failed to satisfy constraint: string attribute");

  if (tblgen_constraints && !((tblgen_constraints.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'constraints' failed to satisfy constraint: string attribute");

  if (tblgen_has_side_effects && !((tblgen_has_side_effects.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'has_side_effects' failed to satisfy constraint: unit attribute");

  if (tblgen_is_align_stack && !((tblgen_is_align_stack.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'is_align_stack' failed to satisfy constraint: unit attribute");

  if (tblgen_asm_dialect && !((tblgen_asm_dialect.isa<::mlir::LLVM::AsmDialectAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'asm_dialect' failed to satisfy constraint: ATT (0) or Intel (1) asm dialect");

  if (tblgen_operand_attrs && !((tblgen_operand_attrs.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'operand_attrs' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InlineAsmOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InlineAsmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InlineAsmOp::getOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange InlineAsmOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InlineAsmOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range InlineAsmOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InlineAsmOp::getRes() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : ::llvm::cast<::mlir::Value>(*results.begin());
}

::mlir::StringAttr InlineAsmOp::getAsmStringAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsmStringAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef InlineAsmOp::getAsmString() {
  auto attr = getAsmStringAttr();
  return attr.getValue();
}

::mlir::StringAttr InlineAsmOp::getConstraintsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getConstraintsAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef InlineAsmOp::getConstraints() {
  auto attr = getConstraintsAttr();
  return attr.getValue();
}

::mlir::UnitAttr InlineAsmOp::getHasSideEffectsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getHasSideEffectsAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool InlineAsmOp::getHasSideEffects() {
  auto attr = getHasSideEffectsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr InlineAsmOp::getIsAlignStackAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getIsAlignStackAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool InlineAsmOp::getIsAlignStack() {
  auto attr = getIsAlignStackAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::AsmDialectAttr InlineAsmOp::getAsmDialectAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAsmDialectAttrName()).dyn_cast_or_null<::mlir::LLVM::AsmDialectAttr>();
}

::std::optional<::mlir::LLVM::AsmDialect> InlineAsmOp::getAsmDialect() {
  auto attr = getAsmDialectAttr();
  return attr ? ::std::optional<::mlir::LLVM::AsmDialect>(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr InlineAsmOp::getOperandAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > InlineAsmOp::getOperandAttrs() {
  auto attr = getOperandAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void InlineAsmOp::setAsmStringAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getAsmStringAttrName(), attr);
}

void InlineAsmOp::setAsmString(::llvm::StringRef attrValue) {
  (*this)->setAttr(getAsmStringAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void InlineAsmOp::setConstraintsAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getConstraintsAttrName(), attr);
}

void InlineAsmOp::setConstraints(::llvm::StringRef attrValue) {
  (*this)->setAttr(getConstraintsAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void InlineAsmOp::setHasSideEffectsAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getHasSideEffectsAttrName(), attr);
}

void InlineAsmOp::setHasSideEffects(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHasSideEffectsAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getHasSideEffectsAttrName());
}

void InlineAsmOp::setIsAlignStackAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIsAlignStackAttrName(), attr);
}

void InlineAsmOp::setIsAlignStack(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIsAlignStackAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getIsAlignStackAttrName());
}

void InlineAsmOp::setAsmDialectAttr(::mlir::LLVM::AsmDialectAttr attr) {
  (*this)->setAttr(getAsmDialectAttrName(), attr);
}

void InlineAsmOp::setAsmDialect(::std::optional<::mlir::LLVM::AsmDialect> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAsmDialectAttrName(), ::mlir::LLVM::AsmDialectAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getAsmDialectAttrName());
}

void InlineAsmOp::setOperandAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOperandAttrsAttrName(), attr);
}

::mlir::Attribute InlineAsmOp::removeHasSideEffectsAttr() {
  return (*this)->removeAttr(getHasSideEffectsAttrName());
}

::mlir::Attribute InlineAsmOp::removeIsAlignStackAttr() {
  return (*this)->removeAttr(getIsAlignStackAttrName());
}

::mlir::Attribute InlineAsmOp::removeAsmDialectAttr() {
  return (*this)->removeAttr(getAsmDialectAttrName());
}

::mlir::Attribute InlineAsmOp::removeOperandAttrsAttr() {
  return (*this)->removeAttr(getOperandAttrsAttrName());
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::mlir::StringAttr asm_string, ::mlir::StringAttr constraints, /*optional*/::mlir::UnitAttr has_side_effects, /*optional*/::mlir::UnitAttr is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), asm_string);
  odsState.addAttribute(getConstraintsAttrName(odsState.name), constraints);
  if (has_side_effects) {
    odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), has_side_effects);
  }
  if (is_align_stack) {
    odsState.addAttribute(getIsAlignStackAttrName(odsState.name), is_align_stack);
  }
  if (asm_dialect) {
    odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
    odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  if (res)
    odsState.addTypes(res);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::StringAttr asm_string, ::mlir::StringAttr constraints, /*optional*/::mlir::UnitAttr has_side_effects, /*optional*/::mlir::UnitAttr is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), asm_string);
  odsState.addAttribute(getConstraintsAttrName(odsState.name), constraints);
  if (has_side_effects) {
    odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), has_side_effects);
  }
  if (is_align_stack) {
    odsState.addAttribute(getIsAlignStackAttrName(odsState.name), is_align_stack);
  }
  if (asm_dialect) {
    odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
    odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  odsState.addTypes(resultTypes);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), odsBuilder.getStringAttr(asm_string));
  odsState.addAttribute(getConstraintsAttrName(odsState.name), odsBuilder.getStringAttr(constraints));
  if (has_side_effects) {
    odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), ((has_side_effects) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (is_align_stack) {
    odsState.addAttribute(getIsAlignStackAttrName(odsState.name), ((is_align_stack) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (asm_dialect) {
    odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
    odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  if (res)
    odsState.addTypes(res);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), odsBuilder.getStringAttr(asm_string));
  odsState.addAttribute(getConstraintsAttrName(odsState.name), odsBuilder.getStringAttr(constraints));
  if (has_side_effects) {
    odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), ((has_side_effects) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (is_align_stack) {
    odsState.addAttribute(getIsAlignStackAttrName(odsState.name), ((is_align_stack) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (asm_dialect) {
    odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
    odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  odsState.addTypes(resultTypes);
}

void InlineAsmOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InlineAsmOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_asm_string;
  ::mlir::Attribute tblgen_asm_dialect;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'asm_string'");
    if (namedAttrIt->getName() == getAsmStringAttrName()) {
      tblgen_asm_string = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsmDialectAttrName()) {
      tblgen_asm_dialect = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_constraints;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'constraints'");
    if (namedAttrIt->getName() == getConstraintsAttrName()) {
      tblgen_constraints = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_has_side_effects;
  ::mlir::Attribute tblgen_is_align_stack;
  ::mlir::Attribute tblgen_operand_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHasSideEffectsAttrName()) {
      tblgen_has_side_effects = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIsAlignStackAttrName()) {
      tblgen_is_align_stack = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOperandAttrsAttrName()) {
      tblgen_operand_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_asm_string, "asm_string")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_constraints, "constraints")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_has_side_effects, "has_side_effects")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_is_align_stack, "is_align_stack")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_asm_dialect, "asm_dialect")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps23(*this, tblgen_operand_attrs, "operand_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InlineAsmOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult InlineAsmOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::LLVM::AsmDialectAttr asm_dialectAttr;
  ::mlir::ArrayAttr operand_attrsAttr;
  ::mlir::StringAttr asm_stringAttr;
  ::mlir::StringAttr constraintsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  if (::mlir::succeeded(parser.parseOptionalKeyword("has_side_effects"))) {
    result.addAttribute("has_side_effects", parser.getBuilder().getUnitAttr());
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("is_align_stack"))) {
    result.addAttribute("is_align_stack", parser.getBuilder().getUnitAttr());
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("asm_dialect"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"att","intel"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "asm_dialect", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'asm_dialect' [att, intel]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAsmDialect(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "asm_dialect attribute specification: \"" << attrStr << '"';;

      asm_dialectAttr = ::mlir::LLVM::AsmDialectAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("asm_dialect", asm_dialectAttr);
    }
  }
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("operand_attrs"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(operand_attrsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "operand_attrs",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(asm_stringAttr, parser.getBuilder().getType<::mlir::NoneType>(), "asm_string",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(constraintsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constraints",
          result.attributes)) {
    return ::mlir::failure();
  }
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InlineAsmOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("has_side_effects")) {
    _odsPrinter << ' ' << "has_side_effects";
  }
  if ((*this)->getAttr("is_align_stack")) {
    _odsPrinter << ' ' << "is_align_stack";
  }
  if ((*this)->getAttr("asm_dialect")) {
    _odsPrinter << ' ' << "asm_dialect";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';

    {
      auto caseValue = *getAsmDialect();
      auto caseValueStr = stringifyAsmDialect(caseValue);
      _odsPrinter << caseValueStr;
    }
  }
  if ((*this)->getAttr("operand_attrs")) {
    _odsPrinter << ' ' << "operand_attrs";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getOperandAttrsAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("has_side_effects");
  elidedAttrs.push_back("is_align_stack");
  elidedAttrs.push_back("asm_dialect");
  elidedAttrs.push_back("operand_attrs");
  elidedAttrs.push_back("asm_string");
  elidedAttrs.push_back("constraints");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHasSideEffectsAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("has_side_effects");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIsAlignStackAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("is_align_stack");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getAsmStringAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getConstraintsAttr());
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InlineAsmOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InsertElementOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InsertElementOpGenericAdaptorBase::InsertElementOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.insertelement", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InsertElementOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr InsertElementOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
InsertElementOpAdaptor::InsertElementOpAdaptor(InsertElementOp op) : InsertElementOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InsertElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::getVector() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value InsertElementOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::TypedValue<::mlir::IntegerType> InsertElementOp::getPosition() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
}

::mlir::MutableOperandRange InsertElementOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::getPositionMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);
  odsState.addTypes(res);
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertElementOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertElementOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(LLVM::getVectorElementType((*this->getODSOperands(0).begin()).getType()), (*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that argument type matches vector element type");
  if (!((((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()) && ((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {res, vector} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertElementOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult InsertElementOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult InsertElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand positionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> positionOperands(positionRawOperands);  ::llvm::SMLoc positionOperandsLoc;
  (void)positionOperandsLoc;
  ::mlir::Type positionRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> positionTypes(positionRawTypes);
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  positionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(positionRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    positionRawTypes[0] = type;
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  for (::mlir::Type type : vectorTypes) {
    (void)type;
    if (!((::mlir::LLVM::isCompatibleVectorType(type)))) {
      return parser.emitError(parser.getNameLoc()) << "'vector' must be LLVM dialect-compatible vector type, but got " << type;
    }
  }
  result.addTypes(vectorTypes[0]);
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, LLVM::getVectorElementType(vectorTypes[0]), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(positionOperands, positionTypes, positionOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertElementOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << "[";
  _odsPrinter << getPosition();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getPosition().getType();
    if (auto validType = type.dyn_cast<::mlir::IntegerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertElementOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InsertElementOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InsertValueOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InsertValueOpGenericAdaptorBase::InsertValueOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.insertvalue", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InsertValueOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr InsertValueOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr InsertValueOpGenericAdaptorBase::getPositionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InsertValueOp::getPositionAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> InsertValueOpGenericAdaptorBase::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

} // namespace detail
InsertValueOpAdaptor::InsertValueOpAdaptor(InsertValueOp op) : InsertValueOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InsertValueOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.insertvalue' op ""requires attribute 'position'");
    if (namedAttrIt->getName() == InsertValueOp::getPositionAttrName(*odsOpName)) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_position && !((tblgen_position.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'llvm.insertvalue' op ""attribute 'position' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertValueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertValueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOp::getContainer() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value InsertValueOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange InsertValueOp::getContainerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertValueOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertValueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertValueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr InsertValueOp::getPositionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPositionAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> InsertValueOp::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

void InsertValueOp::setPositionAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPositionAttrName(), attr);
}

void InsertValueOp::setPosition(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPositionAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::Value value, ::mlir::DenseI64ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  odsState.addTypes(res);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value container, ::mlir::Value value, ::mlir::DenseI64ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertValueOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::Value value, ::mlir::DenseI64ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::Value value, ::llvm::ArrayRef<int64_t> position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(position));
  odsState.addTypes(res);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value container, ::mlir::Value value, ::llvm::ArrayRef<int64_t> position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(position));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertValueOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::Value value, ::llvm::ArrayRef<int64_t> position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(position));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertValueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertValueOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertValueOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'position'");
    if (namedAttrIt->getName() == getPositionAttrName()) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps12(*this, tblgen_position, "position")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {container, res} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertValueOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertValueOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult InsertValueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand containerRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> containerOperands(containerRawOperands);  ::llvm::SMLoc containerOperandsLoc;
  (void)containerOperandsLoc;
  ::mlir::DenseI64ArrayAttr positionAttr;
  ::mlir::Type containerRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> containerTypes(containerRawTypes);
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  containerOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(containerRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(positionAttr, ::mlir::Type{}, "position",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    containerRawTypes[0] = type;
  }
  {
    if (parseInsertExtractValueElementType(parser, valueRawTypes[0], containerRawTypes[0], positionAttr))
      return ::mlir::failure();
  }
  result.addTypes(containerTypes[0]);
  if (parser.resolveOperands(containerOperands, containerTypes, containerOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertValueOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getContainer();
_odsPrinter.printStrippedAttrOrType(getPositionAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("position");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getContainer().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  printInsertExtractValueElementType(_odsPrinter, *this, getValue().getType(), getContainer().getType(), getPositionAttr());
}

void InsertValueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InsertValueOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::IntToPtrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
IntToPtrOpGenericAdaptorBase::IntToPtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.inttoptr", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> IntToPtrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr IntToPtrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
IntToPtrOpAdaptor::IntToPtrOpAdaptor(IntToPtrOp op) : IntToPtrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult IntToPtrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IntToPtrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IntToPtrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange IntToPtrOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IntToPtrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IntToPtrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IntToPtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IntToPtrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IntToPtrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IntToPtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IntToPtrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void IntToPtrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::IntToPtrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InvokeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InvokeOpGenericAdaptorBase::InvokeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.invoke", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InvokeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InvokeOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr InvokeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr InvokeOpGenericAdaptorBase::getCalleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, InvokeOp::getCalleeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > InvokeOpGenericAdaptorBase::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ElementsAttr InvokeOpGenericAdaptorBase::getBranchWeightsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, InvokeOp::getBranchWeightsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::std::optional< ::mlir::ElementsAttr > InvokeOpGenericAdaptorBase::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

} // namespace detail
InvokeOpAdaptor::InvokeOpAdaptor(InvokeOp op) : InvokeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InvokeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_branch_weights;
  ::mlir::Attribute tblgen_callee;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.invoke' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == InvokeOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == InvokeOp::getBranchWeightsAttrName(*odsOpName)) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == InvokeOp::getCalleeAttrName(*odsOpName)) {
      tblgen_callee = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'llvm.invoke' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_callee && !((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.invoke' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_branch_weights && !((tblgen_branch_weights.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.invoke' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InvokeOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range InvokeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InvokeOp::getCalleeOperands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range InvokeOp::getNormalDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range InvokeOp::getUnwindDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InvokeOp::getCalleeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InvokeOp::getNormalDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InvokeOp::getUnwindDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> InvokeOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range InvokeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *InvokeOp::getNormalDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *InvokeOp::getUnwindDest() {
  return (*this)->getSuccessor(1);
}

::mlir::FlatSymbolRefAttr InvokeOp::getCalleeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCalleeAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::std::optional< ::llvm::StringRef > InvokeOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ElementsAttr InvokeOp::getBranchWeightsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBranchWeightsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::std::optional< ::mlir::ElementsAttr > InvokeOp::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

void InvokeOp::setCalleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

void InvokeOp::setCallee(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCalleeAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCalleeAttrName());
}

void InvokeOp::setBranchWeightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getBranchWeightsAttrName(), attr);
}

::mlir::Attribute InvokeOp::removeCalleeAttr() {
  return (*this)->removeAttr(getCalleeAttrName());
}

::mlir::Attribute InvokeOp::removeBranchWeightsAttr() {
  return (*this)->removeAttr(getBranchWeightsAttrName());
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange tys, FlatSymbolRefAttr callee, ValueRange ops, Block*normal, ValueRange normalOps, Block*unwind, ValueRange unwindOps) {
      odsState.addAttribute("callee", callee);
      build(odsBuilder, odsState, tys, ops, normal, normalOps, unwind, unwindOps);
    
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange tys, ValueRange ops, Block*normal, ValueRange normalOps, Block*unwind, ValueRange unwindOps) {
      build(odsBuilder, odsState, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, nullptr, normal, unwind);
    
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange callee_operands, ::mlir::ValueRange normalDestOperands, ::mlir::ValueRange unwindDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *normalDest, ::mlir::Block *unwindDest) {
  odsState.addOperands(callee_operands);
  odsState.addOperands(normalDestOperands);
  odsState.addOperands(unwindDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(callee_operands.size()), static_cast<int32_t>(normalDestOperands.size()), static_cast<int32_t>(unwindDestOperands.size())}));
  if (callee) {
    odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(normalDest);
  odsState.addSuccessors(unwindDest);
  odsState.addTypes(resultType0);
}

void InvokeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InvokeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_branch_weights;
  ::mlir::Attribute tblgen_callee;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBranchWeightsAttrName()) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCalleeAttrName()) {
      tblgen_callee = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_callee, "callee")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps10(*this, tblgen_branch_weights, "branch_weights")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult InvokeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InvokeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LLVMFuncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LLVMFuncOpGenericAdaptorBase::LLVMFuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.func", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LLVMFuncOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LLVMFuncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr LLVMFuncOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LLVMFuncOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef LLVMFuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr LLVMFuncOpGenericAdaptorBase::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LLVMFuncOp::getFunctionTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::LLVM::LLVMFunctionType LLVMFuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::LLVM::LLVMFunctionType>();
}

::mlir::LLVM::LinkageAttr LLVMFuncOpGenericAdaptorBase::getLinkageAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LLVMFuncOp::getLinkageAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::LinkageAttr>();
  return attr;
}

::mlir::LLVM::Linkage LLVMFuncOpGenericAdaptorBase::getLinkage() {
  auto attr = getLinkageAttr();
  return attr.getLinkage();
}

::mlir::UnitAttr LLVMFuncOpGenericAdaptorBase::getDsoLocalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, LLVMFuncOp::getDsoLocalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LLVMFuncOpGenericAdaptorBase::getDsoLocal() {
  auto attr = getDsoLocalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::CConvAttr LLVMFuncOpGenericAdaptorBase::getCConvAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, LLVMFuncOp::getCConvAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::CConvAttr>();
  return attr;
}

::mlir::LLVM::cconv::CConv LLVMFuncOpGenericAdaptorBase::getCConv() {
  auto attr = getCConvAttr();
  return attr.getCallingConv();
}

::mlir::FlatSymbolRefAttr LLVMFuncOpGenericAdaptorBase::getPersonalityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LLVMFuncOp::getPersonalityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > LLVMFuncOpGenericAdaptorBase::getPersonality() {
  auto attr = getPersonalityAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::StringAttr LLVMFuncOpGenericAdaptorBase::getGarbageCollectorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LLVMFuncOp::getGarbageCollectorAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > LLVMFuncOpGenericAdaptorBase::getGarbageCollector() {
  auto attr = getGarbageCollectorAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr LLVMFuncOpGenericAdaptorBase::getPassthroughAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LLVMFuncOp::getPassthroughAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LLVMFuncOpGenericAdaptorBase::getPassthrough() {
  auto attr = getPassthroughAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LLVMFuncOpGenericAdaptorBase::getArgAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, LLVMFuncOp::getArgAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LLVMFuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LLVMFuncOpGenericAdaptorBase::getResAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LLVMFuncOp::getResAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LLVMFuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::IntegerAttr LLVMFuncOpGenericAdaptorBase::getFunctionEntryCountAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, LLVMFuncOp::getFunctionEntryCountAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> LLVMFuncOpGenericAdaptorBase::getFunctionEntryCount() {
  auto attr = getFunctionEntryCountAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::LLVM::MemoryEffectsAttr LLVMFuncOpGenericAdaptorBase::getMemoryAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LLVMFuncOp::getMemoryAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::MemoryEffectsAttr>();
  return attr;
}

::std::optional<::mlir::LLVM::MemoryEffectsAttr> LLVMFuncOpGenericAdaptorBase::getMemory() {
  auto attr = getMemoryAttr();
  return attr ? ::std::optional<::mlir::LLVM::MemoryEffectsAttr>(attr.cast<::mlir::LLVM::MemoryEffectsAttr>()) : (::std::nullopt);
}

::mlir::LLVM::VisibilityAttr LLVMFuncOpGenericAdaptorBase::getVisibility_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, LLVMFuncOp::getVisibility_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::VisibilityAttr>();
  return attr;
}

::mlir::LLVM::Visibility LLVMFuncOpGenericAdaptorBase::getVisibility_() {
  auto attr = getVisibility_Attr();
  return attr.getValue();
}

::mlir::Region &LLVMFuncOpGenericAdaptorBase::getBody() {
  return *odsRegions[0];
}

::mlir::RegionRange LLVMFuncOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
LLVMFuncOpAdaptor::LLVMFuncOpAdaptor(LLVMFuncOp op) : LLVMFuncOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LLVMFuncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_CConv;
  ::mlir::Attribute tblgen_arg_attrs;
  ::mlir::Attribute tblgen_dso_local;
  ::mlir::Attribute tblgen_function_entry_count;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.func' op ""requires attribute 'function_type'");
    if (namedAttrIt->getName() == LLVMFuncOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getCConvAttrName(*odsOpName)) {
      tblgen_CConv = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getArgAttrsAttrName(*odsOpName)) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getDsoLocalAttrName(*odsOpName)) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getFunctionEntryCountAttrName(*odsOpName)) {
      tblgen_function_entry_count = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_garbageCollector;
  ::mlir::Attribute tblgen_linkage;
  ::mlir::Attribute tblgen_memory;
  ::mlir::Attribute tblgen_passthrough;
  ::mlir::Attribute tblgen_personality;
  ::mlir::Attribute tblgen_res_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.func' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == LLVMFuncOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getGarbageCollectorAttrName(*odsOpName)) {
      tblgen_garbageCollector = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getLinkageAttrName(*odsOpName)) {
      tblgen_linkage = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getMemoryAttrName(*odsOpName)) {
      tblgen_memory = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getPassthroughAttrName(*odsOpName)) {
      tblgen_passthrough = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getPersonalityAttrName(*odsOpName)) {
      tblgen_personality = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getResAttrsAttrName(*odsOpName)) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_visibility_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getVisibility_AttrName(*odsOpName)) {
      tblgen_visibility_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::LLVM::LLVMFunctionType>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::LLVM::LLVMFunctionType>()))))
    return emitError(loc, "'llvm.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of LLVM function type");

  if (tblgen_linkage && !((tblgen_linkage.isa<::mlir::LLVM::LinkageAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'linkage' failed to satisfy constraint: LLVM Linkage specification");

  if (tblgen_dso_local && !((tblgen_dso_local.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'dso_local' failed to satisfy constraint: unit attribute");

  if (tblgen_CConv && !((tblgen_CConv.isa<::mlir::LLVM::CConvAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'CConv' failed to satisfy constraint: LLVM Calling Convention specification");

  if (tblgen_personality && !((tblgen_personality.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'personality' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_garbageCollector && !((tblgen_garbageCollector.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'garbageCollector' failed to satisfy constraint: string attribute");

  if (tblgen_passthrough && !((tblgen_passthrough.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'passthrough' failed to satisfy constraint: array attribute");

  if (tblgen_arg_attrs && !(((tblgen_arg_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_arg_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'llvm.func' op ""attribute 'arg_attrs' failed to satisfy constraint: Array of dictionary attributes");

  if (tblgen_res_attrs && !(((tblgen_res_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_res_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'llvm.func' op ""attribute 'res_attrs' failed to satisfy constraint: Array of dictionary attributes");

  if (tblgen_function_entry_count && !(((tblgen_function_entry_count.isa<::mlir::IntegerAttr>())) && ((tblgen_function_entry_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.func' op ""attribute 'function_entry_count' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory && !((tblgen_memory.isa<::mlir::LLVM::MemoryEffectsAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'memory' failed to satisfy constraint: ");

  if (tblgen_visibility_ && !((tblgen_visibility_.isa<::mlir::LLVM::VisibilityAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'visibility_' failed to satisfy constraint: LLVM GlobalValue Visibility");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LLVMFuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LLVMFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LLVMFuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LLVMFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &LLVMFuncOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr LLVMFuncOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef LLVMFuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr LLVMFuncOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::LLVM::LLVMFunctionType LLVMFuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::LLVM::LLVMFunctionType>();
}

::mlir::LLVM::LinkageAttr LLVMFuncOp::getLinkageAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getLinkageAttrName()).dyn_cast_or_null<::mlir::LLVM::LinkageAttr>();
}

::mlir::LLVM::Linkage LLVMFuncOp::getLinkage() {
  auto attr = getLinkageAttr();
  return attr.getLinkage();
}

::mlir::UnitAttr LLVMFuncOp::getDsoLocalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDsoLocalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LLVMFuncOp::getDsoLocal() {
  auto attr = getDsoLocalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::CConvAttr LLVMFuncOp::getCConvAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCConvAttrName()).dyn_cast_or_null<::mlir::LLVM::CConvAttr>();
}

::mlir::LLVM::cconv::CConv LLVMFuncOp::getCConv() {
  auto attr = getCConvAttr();
  return attr.getCallingConv();
}

::mlir::FlatSymbolRefAttr LLVMFuncOp::getPersonalityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPersonalityAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::std::optional< ::llvm::StringRef > LLVMFuncOp::getPersonality() {
  auto attr = getPersonalityAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::StringAttr LLVMFuncOp::getGarbageCollectorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getGarbageCollectorAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > LLVMFuncOp::getGarbageCollector() {
  auto attr = getGarbageCollectorAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr LLVMFuncOp::getPassthroughAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPassthroughAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LLVMFuncOp::getPassthrough() {
  auto attr = getPassthroughAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LLVMFuncOp::getArgAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getArgAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LLVMFuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LLVMFuncOp::getResAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getResAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LLVMFuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::IntegerAttr LLVMFuncOp::getFunctionEntryCountAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getFunctionEntryCountAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> LLVMFuncOp::getFunctionEntryCount() {
  auto attr = getFunctionEntryCountAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::LLVM::MemoryEffectsAttr LLVMFuncOp::getMemoryAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getMemoryAttrName()).dyn_cast_or_null<::mlir::LLVM::MemoryEffectsAttr>();
}

::std::optional<::mlir::LLVM::MemoryEffectsAttr> LLVMFuncOp::getMemory() {
  auto attr = getMemoryAttr();
  return attr ? ::std::optional<::mlir::LLVM::MemoryEffectsAttr>(attr.cast<::mlir::LLVM::MemoryEffectsAttr>()) : (::std::nullopt);
}

::mlir::LLVM::VisibilityAttr LLVMFuncOp::getVisibility_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getVisibility_AttrName()).dyn_cast_or_null<::mlir::LLVM::VisibilityAttr>();
}

::mlir::LLVM::Visibility LLVMFuncOp::getVisibility_() {
  auto attr = getVisibility_Attr();
  return attr.getValue();
}

void LLVMFuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void LLVMFuncOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void LLVMFuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void LLVMFuncOp::setFunctionType(::mlir::LLVM::LLVMFunctionType attrValue) {
  (*this)->setAttr(getFunctionTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void LLVMFuncOp::setLinkageAttr(::mlir::LLVM::LinkageAttr attr) {
  (*this)->setAttr(getLinkageAttrName(), attr);
}

void LLVMFuncOp::setLinkage(::mlir::LLVM::Linkage attrValue) {
  (*this)->setAttr(getLinkageAttrName(), ::mlir::LLVM::LinkageAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void LLVMFuncOp::setDsoLocalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getDsoLocalAttrName(), attr);
}

void LLVMFuncOp::setDsoLocal(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDsoLocalAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getDsoLocalAttrName());
}

void LLVMFuncOp::setCConvAttr(::mlir::LLVM::CConvAttr attr) {
  (*this)->setAttr(getCConvAttrName(), attr);
}

void LLVMFuncOp::setCConv(::mlir::LLVM::cconv::CConv attrValue) {
  (*this)->setAttr(getCConvAttrName(), ::mlir::LLVM::CConvAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void LLVMFuncOp::setPersonalityAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getPersonalityAttrName(), attr);
}

void LLVMFuncOp::setPersonality(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getPersonalityAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getPersonalityAttrName());
}

void LLVMFuncOp::setGarbageCollectorAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getGarbageCollectorAttrName(), attr);
}

void LLVMFuncOp::setGarbageCollector(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getGarbageCollectorAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getGarbageCollectorAttrName());
}

void LLVMFuncOp::setPassthroughAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPassthroughAttrName(), attr);
}

void LLVMFuncOp::setArgAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgAttrsAttrName(), attr);
}

void LLVMFuncOp::setResAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getResAttrsAttrName(), attr);
}

void LLVMFuncOp::setFunctionEntryCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getFunctionEntryCountAttrName(), attr);
}

void LLVMFuncOp::setFunctionEntryCount(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getFunctionEntryCountAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getFunctionEntryCountAttrName());
}

void LLVMFuncOp::setMemoryAttr(::mlir::LLVM::MemoryEffectsAttr attr) {
  (*this)->setAttr(getMemoryAttrName(), attr);
}

void LLVMFuncOp::setVisibility_Attr(::mlir::LLVM::VisibilityAttr attr) {
  (*this)->setAttr(getVisibility_AttrName(), attr);
}

void LLVMFuncOp::setVisibility_(::mlir::LLVM::Visibility attrValue) {
  (*this)->setAttr(getVisibility_AttrName(), ::mlir::LLVM::VisibilityAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

::mlir::Attribute LLVMFuncOp::removeDsoLocalAttr() {
  return (*this)->removeAttr(getDsoLocalAttrName());
}

::mlir::Attribute LLVMFuncOp::removePersonalityAttr() {
  return (*this)->removeAttr(getPersonalityAttrName());
}

::mlir::Attribute LLVMFuncOp::removeGarbageCollectorAttr() {
  return (*this)->removeAttr(getGarbageCollectorAttrName());
}

::mlir::Attribute LLVMFuncOp::removePassthroughAttr() {
  return (*this)->removeAttr(getPassthroughAttrName());
}

::mlir::Attribute LLVMFuncOp::removeArgAttrsAttr() {
  return (*this)->removeAttr(getArgAttrsAttrName());
}

::mlir::Attribute LLVMFuncOp::removeResAttrsAttr() {
  return (*this)->removeAttr(getResAttrsAttrName());
}

::mlir::Attribute LLVMFuncOp::removeFunctionEntryCountAttr() {
  return (*this)->removeAttr(getFunctionEntryCountAttrName());
}

::mlir::Attribute LLVMFuncOp::removeMemoryAttr() {
  return (*this)->removeAttr(getMemoryAttrName());
}

void LLVMFuncOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[6])) {
    attributes.append(attrNames[6], ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), Linkage::External));
  }
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::CConvAttr::get(odsBuilder.getContext(), CConv::C));
  }
  if (!attributes.get(attrNames[12])) {
    attributes.append(attrNames[12], ::mlir::LLVM::VisibilityAttr::get(odsBuilder.getContext(), mlir::LLVM::Visibility::Default));
  }
}

::mlir::LogicalResult LLVMFuncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_CConv;
  ::mlir::Attribute tblgen_arg_attrs;
  ::mlir::Attribute tblgen_dso_local;
  ::mlir::Attribute tblgen_function_entry_count;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCConvAttrName()) {
      tblgen_CConv = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getArgAttrsAttrName()) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDsoLocalAttrName()) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getFunctionEntryCountAttrName()) {
      tblgen_function_entry_count = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_garbageCollector;
  ::mlir::Attribute tblgen_linkage;
  ::mlir::Attribute tblgen_memory;
  ::mlir::Attribute tblgen_passthrough;
  ::mlir::Attribute tblgen_personality;
  ::mlir::Attribute tblgen_res_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getGarbageCollectorAttrName()) {
      tblgen_garbageCollector = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getLinkageAttrName()) {
      tblgen_linkage = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getMemoryAttrName()) {
      tblgen_memory = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPassthroughAttrName()) {
      tblgen_passthrough = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPersonalityAttrName()) {
      tblgen_personality = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getResAttrsAttrName()) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_visibility_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getVisibility_AttrName()) {
      tblgen_visibility_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps24(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps17(*this, tblgen_linkage, "linkage")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_dso_local, "dso_local")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps25(*this, tblgen_CConv, "CConv")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_personality, "personality")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_garbageCollector, "garbageCollector")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps23(*this, tblgen_passthrough, "passthrough")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps26(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps26(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_function_entry_count, "function_entry_count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps27(*this, tblgen_memory, "memory")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps20(*this, tblgen_visibility_, "visibility_")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LLVMOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult LLVMFuncOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMFuncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LShrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LShrOpGenericAdaptorBase::LShrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.lshr", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LShrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LShrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LShrOpAdaptor::LShrOpAdaptor(LShrOp op) : LShrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value LShrOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange LShrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LShrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LShrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(LShrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult LShrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LShrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult LShrOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult LShrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LShrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LShrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LShrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LandingpadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LandingpadOpGenericAdaptorBase::LandingpadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.landingpad", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LandingpadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr LandingpadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr LandingpadOpGenericAdaptorBase::getCleanupAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LandingpadOp::getCleanupAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LandingpadOpGenericAdaptorBase::getCleanup() {
  auto attr = getCleanupAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
LandingpadOpAdaptor::LandingpadOpAdaptor(LandingpadOp op) : LandingpadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LandingpadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cleanup;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LandingpadOp::getCleanupAttrName(*odsOpName)) {
      tblgen_cleanup = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_cleanup && !((tblgen_cleanup.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.landingpad' op ""attribute 'cleanup' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LandingpadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LandingpadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LandingpadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LandingpadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LandingpadOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::UnitAttr LandingpadOp::getCleanupAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCleanupAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LandingpadOp::getCleanup() {
  auto attr = getCleanupAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void LandingpadOp::setCleanupAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCleanupAttrName(), attr);
}

void LandingpadOp::setCleanup(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCleanupAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getCleanupAttrName());
}

::mlir::Attribute LandingpadOp::removeCleanupAttr() {
  return (*this)->removeAttr(getCleanupAttrName());
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, /*optional*/::mlir::UnitAttr cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
    odsState.addAttribute(getCleanupAttrName(odsState.name), cleanup);
  }
  odsState.addTypes(res);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
    odsState.addAttribute(getCleanupAttrName(odsState.name), cleanup);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, /*optional*/bool cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
    odsState.addAttribute(getCleanupAttrName(odsState.name), ((cleanup) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(res);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
    odsState.addAttribute(getCleanupAttrName(odsState.name), ((cleanup) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LandingpadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LandingpadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cleanup;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getCleanupAttrName()) {
      tblgen_cleanup = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_cleanup, "cleanup")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LandingpadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LandingpadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoadOpGenericAdaptorBase::LoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.load", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoadOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> LoadOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr LoadOpGenericAdaptorBase::getVolatile_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getVolatile_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoadOpGenericAdaptorBase::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoadOpGenericAdaptorBase::getNontemporalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getNontemporalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoadOpGenericAdaptorBase::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::AtomicOrderingAttr LoadOpGenericAdaptorBase::getOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getOrderingAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrdering LoadOpGenericAdaptorBase::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr LoadOpGenericAdaptorBase::getSyncscopeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getSyncscopeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > LoadOpGenericAdaptorBase::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOpGenericAdaptorBase::getAccessGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getAccessGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LoadOpGenericAdaptorBase::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOpGenericAdaptorBase::getAliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getAliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LoadOpGenericAdaptorBase::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOpGenericAdaptorBase::getNoaliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getNoaliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LoadOpGenericAdaptorBase::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOpGenericAdaptorBase::getTbaaAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getTbaaAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > LoadOpGenericAdaptorBase::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
LoadOpAdaptor::LoadOpAdaptor(LoadOp op) : LoadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_ordering;
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoadOp::getAccessGroupsAttrName(*odsOpName)) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getAliasScopesAttrName(*odsOpName)) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getNoaliasScopesAttrName(*odsOpName)) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getNontemporalAttrName(*odsOpName)) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getOrderingAttrName(*odsOpName)) {
      tblgen_ordering = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getSyncscopeAttrName(*odsOpName)) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getTbaaAttrName(*odsOpName)) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getVolatile_AttrName(*odsOpName)) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.load' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_volatile_ && !((tblgen_volatile_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.load' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");

  if (tblgen_nontemporal && !((tblgen_nontemporal.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.load' op ""attribute 'nontemporal' failed to satisfy constraint: unit attribute");

  if (tblgen_ordering && !((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.load' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_syncscope && !((tblgen_syncscope.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.load' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");

  if (tblgen_access_groups && !(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alias_scopes && !(((tblgen_alias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_noalias_scopes && !(((tblgen_noalias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_noalias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_tbaa && !(((tblgen_tbaa.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_tbaa.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'tbaa' failed to satisfy constraint: symbol ref array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> LoadOp::getAddr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoadOp::getAddrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::IntegerAttr LoadOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> LoadOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr LoadOp::getVolatile_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVolatile_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr LoadOp::getNontemporalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNontemporalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::AtomicOrderingAttr LoadOp::getOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOrderingAttrName()).dyn_cast_or_null<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering LoadOp::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr LoadOp::getSyncscopeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSyncscopeAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > LoadOp::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOp::getAccessGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAccessGroupsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LoadOp::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOp::getAliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LoadOp::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOp::getNoaliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNoaliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LoadOp::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr LoadOp::getTbaaAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTbaaAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > LoadOp::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void LoadOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void LoadOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

void LoadOp::setVolatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getVolatile_AttrName(), attr);
}

void LoadOp::setVolatile_(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getVolatile_AttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getVolatile_AttrName());
}

void LoadOp::setNontemporalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNontemporalAttrName(), attr);
}

void LoadOp::setNontemporal(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNontemporalAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNontemporalAttrName());
}

void LoadOp::setOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getOrderingAttrName(), attr);
}

void LoadOp::setOrdering(::mlir::LLVM::AtomicOrdering attrValue) {
  (*this)->setAttr(getOrderingAttrName(), ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void LoadOp::setSyncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSyncscopeAttrName(), attr);
}

void LoadOp::setSyncscope(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSyncscopeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSyncscopeAttrName());
}

void LoadOp::setAccessGroupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAccessGroupsAttrName(), attr);
}

void LoadOp::setAliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAliasScopesAttrName(), attr);
}

void LoadOp::setNoaliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNoaliasScopesAttrName(), attr);
}

void LoadOp::setTbaaAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTbaaAttrName(), attr);
}

::mlir::Attribute LoadOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute LoadOp::removeVolatile_Attr() {
  return (*this)->removeAttr(getVolatile_AttrName());
}

::mlir::Attribute LoadOp::removeNontemporalAttr() {
  return (*this)->removeAttr(getNontemporalAttrName());
}

::mlir::Attribute LoadOp::removeSyncscopeAttr() {
  return (*this)->removeAttr(getSyncscopeAttrName());
}

::mlir::Attribute LoadOp::removeAccessGroupsAttr() {
  return (*this)->removeAttr(getAccessGroupsAttrName());
}

::mlir::Attribute LoadOp::removeAliasScopesAttr() {
  return (*this)->removeAttr(getAliasScopesAttrName());
}

::mlir::Attribute LoadOp::removeNoaliasScopesAttr() {
  return (*this)->removeAttr(getNoaliasScopesAttrName());
}

::mlir::Attribute LoadOp::removeTbaaAttr() {
  return (*this)->removeAttr(getTbaaAttrName());
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  if (ordering) {
    odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  }
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  if (ordering) {
    odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  }
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), ((nontemporal) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), ((nontemporal) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LoadOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[5])) {
    attributes.append(attrNames[5], ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), AtomicOrdering::not_atomic));
  }
}

::mlir::LogicalResult LoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_ordering;
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAccessGroupsAttrName()) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAliasScopesAttrName()) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNoaliasScopesAttrName()) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNontemporalAttrName()) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOrderingAttrName()) {
      tblgen_ordering = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSyncscopeAttrName()) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTbaaAttrName()) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVolatile_AttrName()) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_volatile_, "volatile_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_nontemporal, "nontemporal")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_ordering, "ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_syncscope, "syncscope")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_access_groups, "access_groups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_alias_scopes, "alias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_noalias_scopes, "noalias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_tbaa, "tbaa")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(addrRawOperands);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::StringAttr syncscopeAttr;
  ::mlir::LLVM::AtomicOrderingAttr orderingAttr;
  ::mlir::Type addrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> addrTypes(addrRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (::mlir::succeeded(parser.parseOptionalKeyword("volatile"))) {
    result.addAttribute("volatile_", parser.getBuilder().getUnitAttr());
  }

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("atomic"))) {
  if (::mlir::succeeded(parser.parseOptionalKeyword("syncscope"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(syncscopeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "syncscope",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"not_atomic","unordered","monotonic","acquire","release","acq_rel","seq_cst"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "ordering", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'ordering' [not_atomic, unordered, monotonic, acquire, release, acq_rel, seq_cst]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicOrdering(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "ordering attribute specification: \"" << attrStr << '"';;

      orderingAttr = ::mlir::LLVM::AtomicOrderingAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("ordering", orderingAttr);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  {
    if (parseLoadType(parser, addrRawTypes[0], resRawTypes[0]))
      return ::mlir::failure();
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("volatile_")) {
    _odsPrinter << ' ' << "volatile";
  }
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  if ((*this)->getAttr("ordering") != ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::OpBuilder((*this)->getContext()).getContext(), AtomicOrdering::not_atomic)) {
    _odsPrinter << ' ' << "atomic";
    if ((*this)->getAttr("syncscope")) {
      _odsPrinter << ' ' << "syncscope";
      _odsPrinter << "(";
      _odsPrinter.printAttributeWithoutType(getSyncscopeAttr());
      _odsPrinter << ")";
    }
    _odsPrinter << ' ';

    {
      auto caseValue = getOrdering();
      auto caseValueStr = stringifyAtomicOrdering(caseValue);
      _odsPrinter << caseValueStr;
    }
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("volatile_");
  elidedAttrs.push_back("syncscope");
  elidedAttrs.push_back("ordering");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getVolatile_Attr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("volatile_");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNontemporalAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nontemporal");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getOrderingAttr();
     if(attr && (attr == ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), AtomicOrdering::not_atomic)))
       elidedAttrs.push_back("ordering");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  printLoadType(_odsPrinter, *this, getAddr().getType(), getRes().getType());
}

void LoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LoadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MetadataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MetadataOpGenericAdaptorBase::MetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.metadata", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MetadataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MetadataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr MetadataOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef MetadataOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::Region &MetadataOpGenericAdaptorBase::getBody() {
  return *odsRegions[0];
}

::mlir::RegionRange MetadataOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MetadataOpAdaptor::MetadataOpAdaptor(MetadataOp op) : MetadataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.metadata' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == MetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.metadata' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> MetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &MetadataOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr MetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef MetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

void MetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void MetadataOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

::mlir::LogicalResult MetadataOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LLVMOps1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MulOpGenericAdaptorBase::MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
MulOpAdaptor::MulOpAdaptor(MulOp op) : MulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value MulOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange MulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MulOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult MulOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult MulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::NullOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
NullOpGenericAdaptorBase::NullOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.null", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> NullOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr NullOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
NullOpAdaptor::NullOpAdaptor(NullOp op) : NullOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult NullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NullOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NullOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> NullOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NullOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> NullOp::getRes() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSResults(0).begin());
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NullOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NullOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult NullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void NullOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getRes().getType();
}

void NullOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::NullOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::OrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
OrOpGenericAdaptorBase::OrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.or", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> OrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr OrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
OrOpAdaptor::OrOpAdaptor(OrOp op) : OrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult OrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value OrOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange OrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(OrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(OrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult OrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult OrOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult OrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void OrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::OrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PoisonOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoisonOpGenericAdaptorBase::PoisonOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.poison", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoisonOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PoisonOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
PoisonOpAdaptor::PoisonOpAdaptor(PoisonOp op) : PoisonOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoisonOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoisonOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoisonOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> PoisonOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoisonOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoisonOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void PoisonOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void PoisonOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void PoisonOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoisonOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoisonOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoisonOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PoisonOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void PoisonOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void PoisonOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::PoisonOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PtrToIntOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PtrToIntOpGenericAdaptorBase::PtrToIntOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.ptrtoint", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PtrToIntOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PtrToIntOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
PtrToIntOpAdaptor::PtrToIntOpAdaptor(PtrToIntOp op) : PtrToIntOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PtrToIntOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PtrToIntOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PtrToIntOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange PtrToIntOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PtrToIntOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PtrToIntOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PtrToIntOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PtrToIntOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PtrToIntOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PtrToIntOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PtrToIntOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void PtrToIntOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::PtrToIntOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ResumeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ResumeOpGenericAdaptorBase::ResumeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.resume", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ResumeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ResumeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ResumeOpAdaptor::ResumeOpAdaptor(ResumeOp op) : ResumeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ResumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ResumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResumeOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ResumeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ResumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void ResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResumeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ResumeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ResumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResumeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ResumeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ReturnOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReturnOpGenericAdaptorBase::ReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.return", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReturnOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ReturnOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp op) : ReturnOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReturnOp::getArg() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::MutableOperandRange ReturnOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange args) {
      build(odsBuilder, odsState, TypeRange(), args);
    
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value arg) {
  if (arg)
    odsState.addOperands(arg);
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value arg) {
  if (arg)
    odsState.addOperands(arg);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argOperands;
  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  {
    argOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      argOperands.push_back(operand);
    }
  }
  if (!argOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      argTypes.push_back(optionalType);
    }
  }
  }
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (getArg()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getArg())
      _odsPrinter << value;
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << (getArg() ? ::llvm::ArrayRef<::mlir::Type>(getArg().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
}

void ReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ReturnOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SDivOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SDivOpGenericAdaptorBase::SDivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sdiv", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SDivOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SDivOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SDivOpAdaptor::SDivOpAdaptor(SDivOp op) : SDivOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value SDivOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange SDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SDivOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SExtOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SExtOpGenericAdaptorBase::SExtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sext", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SExtOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SExtOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SExtOpAdaptor::SExtOpAdaptor(SExtOp op) : SExtOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SExtOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SExtOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SIToFPOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SIToFPOpGenericAdaptorBase::SIToFPOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sitofp", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SIToFPOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SIToFPOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SIToFPOpAdaptor::SIToFPOpAdaptor(SIToFPOp op) : SIToFPOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SIToFPOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SIToFPOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SIToFPOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SIToFPOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SIToFPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SIToFPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SRemOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SRemOpGenericAdaptorBase::SRemOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.srem", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SRemOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SRemOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SRemOpAdaptor::SRemOpAdaptor(SRemOp op) : SRemOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SRemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value SRemOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange SRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SRemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SRemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SRemOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SRemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SRemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SelectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SelectOpGenericAdaptorBase::SelectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.select", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SelectOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SelectOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FastmathFlagsAttr SelectOpGenericAdaptorBase::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SelectOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
  return attr;
}

::mlir::LLVM::FastmathFlags SelectOpGenericAdaptorBase::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

} // namespace detail
SelectOpAdaptor::SelectOpAdaptor(SelectOp op) : SelectOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SelectOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SelectOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FastmathFlagsAttr>())))
    return emitError(loc, "'llvm.select' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::getCondition() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value SelectOp::getTrueValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value SelectOp::getFalseValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
}

::mlir::MutableOperandRange SelectOp::getConditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SelectOp::getTrueValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SelectOp::getFalseValueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LLVM::FastmathFlagsAttr SelectOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FastmathFlagsAttr>();
}

::mlir::LLVM::FastmathFlags SelectOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
  return attr.getValue();
}

void SelectOp::setFastmathFlagsAttr(::mlir::LLVM::FastmathFlagsAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void SelectOp::setFastmathFlags(::mlir::LLVM::FastmathFlags attrValue) {
  (*this)->setAttr(getFastmathFlagsAttrName(), ::mlir::LLVM::FastmathFlagsAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue, ::mlir::LLVM::FastmathFlagsAttr fastmathFlags) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  if (fastmathFlags) {
    odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

void SelectOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {}));
  }
}

::mlir::LogicalResult SelectOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()) && ((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {trueValue, falseValue, res} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult SelectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SelectOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[2].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::mlir::Type conditionRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> conditionTypes(conditionRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    conditionRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const ::mlir::Type>(::llvm::ArrayRef<::mlir::Type>(conditionTypes), ::llvm::ArrayRef<::mlir::Type>(resTypes[0]), ::llvm::ArrayRef<::mlir::Type>(resTypes[0])), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SelectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getFastmathFlagsAttr();
     if(attr && (attr == ::mlir::LLVM::FastmathFlagsAttr::get(odsBuilder.getContext(), {})))
       elidedAttrs.push_back("fastmathFlags");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getCondition().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SelectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SelectOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ShlOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ShlOpGenericAdaptorBase::ShlOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.shl", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ShlOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ShlOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ShlOpAdaptor::ShlOpAdaptor(ShlOp op) : ShlOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ShlOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShlOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShlOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value ShlOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ShlOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShlOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShlOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShlOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ShlOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShlOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ShlOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ShlOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShlOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ShlOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ShlOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShlOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ShlOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ShlOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ShuffleVectorOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ShuffleVectorOpGenericAdaptorBase::ShuffleVectorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.shufflevector", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ShuffleVectorOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ShuffleVectorOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI32ArrayAttr ShuffleVectorOpGenericAdaptorBase::getMaskAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ShuffleVectorOp::getMaskAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int32_t> ShuffleVectorOpGenericAdaptorBase::getMask() {
  auto attr = getMaskAttr();
  return attr;
}

} // namespace detail
ShuffleVectorOpAdaptor::ShuffleVectorOpAdaptor(ShuffleVectorOp op) : ShuffleVectorOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ShuffleVectorOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_mask;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.shufflevector' op ""requires attribute 'mask'");
    if (namedAttrIt->getName() == ShuffleVectorOp::getMaskAttrName(*odsOpName)) {
      tblgen_mask = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_mask && !((tblgen_mask.isa<::mlir::DenseI32ArrayAttr>())))
    return emitError(loc, "'llvm.shufflevector' op ""attribute 'mask' failed to satisfy constraint: i32 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShuffleVectorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShuffleVectorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOp::getV1() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value ShuffleVectorOp::getV2() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ShuffleVectorOp::getV1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleVectorOp::getV2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShuffleVectorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShuffleVectorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::DenseI32ArrayAttr ShuffleVectorOp::getMaskAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMaskAttrName()).cast<::mlir::DenseI32ArrayAttr>();
}

::llvm::ArrayRef<int32_t> ShuffleVectorOp::getMask() {
  auto attr = getMaskAttr();
  return attr;
}

void ShuffleVectorOp::setMaskAttr(::mlir::DenseI32ArrayAttr attr) {
  (*this)->setAttr(getMaskAttrName(), attr);
}

void ShuffleVectorOp::setMask(::llvm::ArrayRef<int32_t> attrValue) {
  (*this)->setAttr(getMaskAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue));
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value v1, ::mlir::Value v2, ::mlir::DenseI32ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);
  odsState.addTypes(res);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value v1, ::mlir::Value v2, ::mlir::DenseI32ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value v1, ::mlir::Value v2, ::llvm::ArrayRef<int32_t> mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr(mask));
  odsState.addTypes(res);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value v1, ::mlir::Value v2, ::llvm::ArrayRef<int32_t> mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr(mask));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShuffleVectorOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_mask;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'mask'");
    if (namedAttrIt->getName() == getMaskAttrName()) {
      tblgen_mask = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_mask, "mask")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()) && ((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {v1, v2} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult ShuffleVectorOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ShuffleVectorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand v1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> v1Operands(v1RawOperands);  ::llvm::SMLoc v1OperandsLoc;
  (void)v1OperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand v2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> v2Operands(v2RawOperands);  ::llvm::SMLoc v2OperandsLoc;
  (void)v2OperandsLoc;
  ::mlir::DenseI32ArrayAttr maskAttr;
  ::mlir::Type v1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> v1Types(v1RawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  v1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(v1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  v2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(v2RawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(maskAttr, ::mlir::Type{}, "mask",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    v1RawTypes[0] = type;
  }
  {
    if (parseShuffleType(parser, v1RawTypes[0], resRawTypes[0], maskAttr))
      return ::mlir::failure();
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(v1Operands, v1Types, v1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(v2Operands, v1Types[0], v2OperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShuffleVectorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getV1();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getV2();
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getMaskAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("mask");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getV1().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  printShuffleType(_odsPrinter, *this, getV1().getType(), getRes().getType(), getMaskAttr());
}

void ShuffleVectorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ShuffleVectorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StoreOpGenericAdaptorBase::StoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.store", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> StoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr StoreOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr StoreOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> StoreOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr StoreOpGenericAdaptorBase::getVolatile_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getVolatile_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool StoreOpGenericAdaptorBase::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr StoreOpGenericAdaptorBase::getNontemporalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getNontemporalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool StoreOpGenericAdaptorBase::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::AtomicOrderingAttr StoreOpGenericAdaptorBase::getOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getOrderingAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrdering StoreOpGenericAdaptorBase::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr StoreOpGenericAdaptorBase::getSyncscopeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getSyncscopeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > StoreOpGenericAdaptorBase::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOpGenericAdaptorBase::getAccessGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getAccessGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StoreOpGenericAdaptorBase::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOpGenericAdaptorBase::getAliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getAliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StoreOpGenericAdaptorBase::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOpGenericAdaptorBase::getNoaliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getNoaliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StoreOpGenericAdaptorBase::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOpGenericAdaptorBase::getTbaaAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getTbaaAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StoreOpGenericAdaptorBase::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StoreOpAdaptor::StoreOpAdaptor(StoreOp op) : StoreOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_ordering;
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StoreOp::getAccessGroupsAttrName(*odsOpName)) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getAliasScopesAttrName(*odsOpName)) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getNoaliasScopesAttrName(*odsOpName)) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getNontemporalAttrName(*odsOpName)) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getOrderingAttrName(*odsOpName)) {
      tblgen_ordering = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getSyncscopeAttrName(*odsOpName)) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getTbaaAttrName(*odsOpName)) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getVolatile_AttrName(*odsOpName)) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.store' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_volatile_ && !((tblgen_volatile_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.store' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");

  if (tblgen_nontemporal && !((tblgen_nontemporal.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.store' op ""attribute 'nontemporal' failed to satisfy constraint: unit attribute");

  if (tblgen_ordering && !((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.store' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_syncscope && !((tblgen_syncscope.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.store' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");

  if (tblgen_access_groups && !(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alias_scopes && !(((tblgen_alias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_noalias_scopes && !(((tblgen_noalias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_noalias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_tbaa && !(((tblgen_tbaa.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_tbaa.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'tbaa' failed to satisfy constraint: symbol ref array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> StoreOp::getAddr() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange StoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::getAddrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr StoreOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> StoreOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::UnitAttr StoreOp::getVolatile_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVolatile_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool StoreOp::getVolatile_() {
  auto attr = getVolatile_Attr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr StoreOp::getNontemporalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNontemporalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool StoreOp::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LLVM::AtomicOrderingAttr StoreOp::getOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOrderingAttrName()).dyn_cast_or_null<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering StoreOp::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr StoreOp::getSyncscopeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSyncscopeAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > StoreOp::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOp::getAccessGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAccessGroupsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > StoreOp::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOp::getAliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > StoreOp::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOp::getNoaliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNoaliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > StoreOp::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr StoreOp::getTbaaAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTbaaAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > StoreOp::getTbaa() {
  auto attr = getTbaaAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StoreOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void StoreOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

void StoreOp::setVolatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getVolatile_AttrName(), attr);
}

void StoreOp::setVolatile_(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getVolatile_AttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getVolatile_AttrName());
}

void StoreOp::setNontemporalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNontemporalAttrName(), attr);
}

void StoreOp::setNontemporal(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNontemporalAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNontemporalAttrName());
}

void StoreOp::setOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getOrderingAttrName(), attr);
}

void StoreOp::setOrdering(::mlir::LLVM::AtomicOrdering attrValue) {
  (*this)->setAttr(getOrderingAttrName(), ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void StoreOp::setSyncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSyncscopeAttrName(), attr);
}

void StoreOp::setSyncscope(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSyncscopeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSyncscopeAttrName());
}

void StoreOp::setAccessGroupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAccessGroupsAttrName(), attr);
}

void StoreOp::setAliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAliasScopesAttrName(), attr);
}

void StoreOp::setNoaliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNoaliasScopesAttrName(), attr);
}

void StoreOp::setTbaaAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTbaaAttrName(), attr);
}

::mlir::Attribute StoreOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute StoreOp::removeVolatile_Attr() {
  return (*this)->removeAttr(getVolatile_AttrName());
}

::mlir::Attribute StoreOp::removeNontemporalAttr() {
  return (*this)->removeAttr(getNontemporalAttrName());
}

::mlir::Attribute StoreOp::removeSyncscopeAttr() {
  return (*this)->removeAttr(getSyncscopeAttrName());
}

::mlir::Attribute StoreOp::removeAccessGroupsAttr() {
  return (*this)->removeAttr(getAccessGroupsAttrName());
}

::mlir::Attribute StoreOp::removeAliasScopesAttr() {
  return (*this)->removeAttr(getAliasScopesAttrName());
}

::mlir::Attribute StoreOp::removeNoaliasScopesAttr() {
  return (*this)->removeAttr(getNoaliasScopesAttrName());
}

::mlir::Attribute StoreOp::removeTbaaAttr() {
  return (*this)->removeAttr(getTbaaAttrName());
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  if (ordering) {
    odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  }
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal, ::mlir::LLVM::AtomicOrderingAttr ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  if (ordering) {
    odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  }
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), ((nontemporal) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal, ::mlir::LLVM::AtomicOrdering ordering, /*optional*/::mlir::StringAttr syncscope, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::ArrayAttr tbaa) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
    odsState.addAttribute(getVolatile_AttrName(odsState.name), ((volatile_) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), ((nontemporal) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  if (syncscope) {
    odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  }
  if (access_groups) {
    odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
    odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
    odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (tbaa) {
    odsState.addAttribute(getTbaaAttrName(odsState.name), tbaa);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void StoreOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[5])) {
    attributes.append(attrNames[5], ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), AtomicOrdering::not_atomic));
  }
}

::mlir::LogicalResult StoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_ordering;
  ::mlir::Attribute tblgen_syncscope;
  ::mlir::Attribute tblgen_tbaa;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAccessGroupsAttrName()) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAliasScopesAttrName()) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNoaliasScopesAttrName()) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNontemporalAttrName()) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOrderingAttrName()) {
      tblgen_ordering = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSyncscopeAttrName()) {
      tblgen_syncscope = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTbaaAttrName()) {
      tblgen_tbaa = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVolatile_AttrName()) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_volatile_, "volatile_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_nontemporal, "nontemporal")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_ordering, "ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_syncscope, "syncscope")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_access_groups, "access_groups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_alias_scopes, "alias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_noalias_scopes, "noalias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_tbaa, "tbaa")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(addrRawOperands);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::StringAttr syncscopeAttr;
  ::mlir::LLVM::AtomicOrderingAttr orderingAttr;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type addrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> addrTypes(addrRawTypes);
  if (::mlir::succeeded(parser.parseOptionalKeyword("volatile"))) {
    result.addAttribute("volatile_", parser.getBuilder().getUnitAttr());
  }

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("atomic"))) {
  if (::mlir::succeeded(parser.parseOptionalKeyword("syncscope"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(syncscopeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "syncscope",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"not_atomic","unordered","monotonic","acquire","release","acq_rel","seq_cst"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "ordering", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'ordering' [not_atomic, unordered, monotonic, acquire, release, acq_rel, seq_cst]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAtomicOrdering(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "ordering attribute specification: \"" << attrStr << '"';;

      orderingAttr = ::mlir::LLVM::AtomicOrderingAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("ordering", orderingAttr);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  {
    if (parseStoreType(parser, valueRawTypes[0], addrRawTypes[0]))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("volatile_")) {
    _odsPrinter << ' ' << "volatile";
  }
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  if ((*this)->getAttr("ordering") != ::mlir::LLVM::AtomicOrderingAttr::get(::mlir::OpBuilder((*this)->getContext()).getContext(), AtomicOrdering::not_atomic)) {
    _odsPrinter << ' ' << "atomic";
    if ((*this)->getAttr("syncscope")) {
      _odsPrinter << ' ' << "syncscope";
      _odsPrinter << "(";
      _odsPrinter.printAttributeWithoutType(getSyncscopeAttr());
      _odsPrinter << ")";
    }
    _odsPrinter << ' ';

    {
      auto caseValue = getOrdering();
      auto caseValueStr = stringifyAtomicOrdering(caseValue);
      _odsPrinter << caseValueStr;
    }
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("volatile_");
  elidedAttrs.push_back("syncscope");
  elidedAttrs.push_back("ordering");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getVolatile_Attr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("volatile_");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNontemporalAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nontemporal");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getOrderingAttr();
     if(attr && (attr == ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), AtomicOrdering::not_atomic)))
       elidedAttrs.push_back("ordering");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  printStoreType(_odsPrinter, *this, getValue().getType(), getAddr().getType());
}

void StoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::StoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SubOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SubOpGenericAdaptorBase::SubOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sub", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SubOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SubOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SubOpAdaptor::SubOpAdaptor(SubOp op) : SubOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SubOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value SubOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange SubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SubOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SubOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SubOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SubOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SwitchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOpGenericAdaptorBase::SwitchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.switch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SwitchOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SwitchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr SwitchOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, SwitchOp::getCaseValuesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::std::optional< ::mlir::ElementsAttr > SwitchOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

::mlir::DenseI32ArrayAttr SwitchOpGenericAdaptorBase::getCaseOperandSegmentsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SwitchOp::getCaseOperandSegmentsAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int32_t> SwitchOpGenericAdaptorBase::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

::mlir::ElementsAttr SwitchOpGenericAdaptorBase::getBranchWeightsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, SwitchOp::getBranchWeightsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::std::optional< ::mlir::ElementsAttr > SwitchOpGenericAdaptorBase::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

} // namespace detail
SwitchOpAdaptor::SwitchOpAdaptor(SwitchOp op) : SwitchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_case_operand_segments;
  ::mlir::Attribute tblgen_branch_weights;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.switch' op ""requires attribute 'case_operand_segments'");
    if (namedAttrIt->getName() == SwitchOp::getCaseOperandSegmentsAttrName(*odsOpName)) {
      tblgen_case_operand_segments = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SwitchOp::getBranchWeightsAttrName(*odsOpName)) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_case_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.switch' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SwitchOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SwitchOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_case_values = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'llvm.switch' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_case_values && !((tblgen_case_values.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.switch' op ""attribute 'case_values' failed to satisfy constraint: constant vector/tensor attribute");

  if (tblgen_case_operand_segments && !((tblgen_case_operand_segments.isa<::mlir::DenseI32ArrayAttr>())))
    return emitError(loc, "'llvm.switch' op ""attribute 'case_operand_segments' failed to satisfy constraint: i32 dense array attribute");

  if (tblgen_branch_weights && !((tblgen_branch_weights.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.switch' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SwitchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> SwitchOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range SwitchOp::getDefaultOperands() {
  return getODSOperands(1);
}

::mlir::OperandRangeRange SwitchOp::getCaseOperands() {
  return getODSOperands(2).split(getCaseOperandSegmentsAttr());
}

::mlir::MutableOperandRange SwitchOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SwitchOp::getDefaultOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRangeRange SwitchOp::getCaseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange.split(*(*this)->getAttrDictionary().getNamed(getCaseOperandSegmentsAttrName()));
}

std::pair<unsigned, unsigned> SwitchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOp::getDefaultDestination() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOp::getCaseDestinations() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ElementsAttr SwitchOp::getCaseValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getCaseValuesAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::std::optional< ::mlir::ElementsAttr > SwitchOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

::mlir::DenseI32ArrayAttr SwitchOp::getCaseOperandSegmentsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCaseOperandSegmentsAttrName()).cast<::mlir::DenseI32ArrayAttr>();
}

::llvm::ArrayRef<int32_t> SwitchOp::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

::mlir::ElementsAttr SwitchOp::getBranchWeightsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBranchWeightsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::std::optional< ::mlir::ElementsAttr > SwitchOp::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::std::optional< ::mlir::ElementsAttr >(attr) : (::std::nullopt);
}

void SwitchOp::setCaseValuesAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOp::setCaseOperandSegmentsAttr(::mlir::DenseI32ArrayAttr attr) {
  (*this)->setAttr(getCaseOperandSegmentsAttrName(), attr);
}

void SwitchOp::setCaseOperandSegments(::llvm::ArrayRef<int32_t> attrValue) {
  (*this)->setAttr(getCaseOperandSegmentsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue));
}

void SwitchOp::setBranchWeightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getBranchWeightsAttrName(), attr);
}

::mlir::Attribute SwitchOp::removeCaseValuesAttr() {
  return (*this)->removeAttr(getCaseValuesAttrName());
}

::mlir::Attribute SwitchOp::removeBranchWeightsAttr() {
  return (*this)->removeAttr(getBranchWeightsAttrName());
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::ElementsAttr case_values, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(value);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    odsState.addAttribute(getCaseOperandSegmentsAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr(rangeSegments));  }
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + range.size(); }))}));
  if (case_values) {
    odsState.addAttribute(getCaseValuesAttrName(odsState.name), case_values);
  }
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::ElementsAttr case_values, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(value);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    odsState.addAttribute(getCaseOperandSegmentsAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr(rangeSegments));  }
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + range.size(); }))}));
  if (case_values) {
    odsState.addAttribute(getCaseValuesAttrName(odsState.name), case_values);
  }
  if (branch_weights) {
    odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_case_operand_segments;
  ::mlir::Attribute tblgen_branch_weights;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'case_operand_segments'");
    if (namedAttrIt->getName() == getCaseOperandSegmentsAttrName()) {
      tblgen_case_operand_segments = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBranchWeightsAttrName()) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_case_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_case_values = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps10(*this, tblgen_case_values, "case_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_case_operand_segments, "case_operand_segments")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps10(*this, tblgen_branch_weights, "branch_weights")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);
    if (::mlir::failed(::mlir::OpTrait::impl::verifyValueSizeAttr(*this, "case_operand_segments", "caseOperands", valueGroup2.size())))
      return ::mlir::failure();

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Block *defaultDestinationSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> defaultOperandsOperands;
  ::llvm::SMLoc defaultOperandsOperandsLoc;
  (void)defaultOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> defaultOperandsTypes;
  ::mlir::ElementsAttr case_valuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> caseDestinationsSuccessors;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> caseOperandsOperands;
    llvm::SmallVector<int32_t> caseOperandsOperandGroupSizes;
  ::llvm::SMLoc caseOperandsOperandsLoc;
  (void)caseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> caseOperandsTypes;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestinationSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  defaultOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(defaultOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(defaultOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    caseOperandsOperandsLoc = parser.getCurrentLocation();
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand>> caseOperandsOperandGroups;
    ::llvm::SmallVector<llvm::SmallVector<::mlir::Type>> caseOperandsTypeGroups;
    if (parseSwitchOpCases(parser, valueRawTypes[0], case_valuesAttr, caseDestinationsSuccessors, caseOperandsOperandGroups, caseOperandsTypeGroups))
      return ::mlir::failure();
    if (case_valuesAttr)
      result.addAttribute("case_values", case_valuesAttr);
    for (const auto &subRange : caseOperandsOperandGroups) {
      caseOperandsOperands.append(subRange.begin(), subRange.end());
      caseOperandsOperandGroupSizes.push_back(subRange.size());
    }
    for (const auto &subRange : caseOperandsTypeGroups)
      caseOperandsTypes.append(subRange.begin(), subRange.end());
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(defaultDestinationSuccessor);
  result.addSuccessors(caseDestinationsSuccessors);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(defaultOperandsOperands.size()), static_cast<int32_t>(caseOperandsOperands.size())}));
  result.addAttribute("case_operand_segments", parser.getBuilder().getDenseI32ArrayAttr(caseOperandsOperandGroupSizes));
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(defaultOperandsOperands, defaultOperandsTypes, defaultOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(caseOperandsOperands, caseOperandsTypes, caseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::IntegerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDestination();
  if (!getDefaultOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getDefaultOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getDefaultOperands().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "[";
  _odsPrinter.printNewline();
  printSwitchOpCases(_odsPrinter, *this, getValue().getType(), getCaseValuesAttr(), getCaseDestinations(), getCaseOperands(), getCaseOperands().getTypes());
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("case_operand_segments");
  elidedAttrs.push_back("case_values");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void SwitchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SwitchOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::TBAARootMetadataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TBAARootMetadataOpGenericAdaptorBase::TBAARootMetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.tbaa_root", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TBAARootMetadataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TBAARootMetadataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr TBAARootMetadataOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TBAARootMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef TBAARootMetadataOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr TBAARootMetadataOpGenericAdaptorBase::getIdentityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TBAARootMetadataOp::getIdentityAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef TBAARootMetadataOpGenericAdaptorBase::getIdentity() {
  auto attr = getIdentityAttr();
  return attr.getValue();
}

} // namespace detail
TBAARootMetadataOpAdaptor::TBAARootMetadataOpAdaptor(TBAARootMetadataOp op) : TBAARootMetadataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TBAARootMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_identity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_root' op ""requires attribute 'identity'");
    if (namedAttrIt->getName() == TBAARootMetadataOp::getIdentityAttrName(*odsOpName)) {
      tblgen_identity = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_root' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == TBAARootMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.tbaa_root' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_identity && !((tblgen_identity.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.tbaa_root' op ""attribute 'identity' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TBAARootMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TBAARootMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TBAARootMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TBAARootMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr TBAARootMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef TBAARootMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr TBAARootMetadataOp::getIdentityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIdentityAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef TBAARootMetadataOp::getIdentity() {
  auto attr = getIdentityAttr();
  return attr.getValue();
}

void TBAARootMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void TBAARootMetadataOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void TBAARootMetadataOp::setIdentityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getIdentityAttrName(), attr);
}

void TBAARootMetadataOp::setIdentity(::llvm::StringRef attrValue) {
  (*this)->setAttr(getIdentityAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void TBAARootMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::StringAttr identity) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getIdentityAttrName(odsState.name), identity);
}

void TBAARootMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::StringAttr identity) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getIdentityAttrName(odsState.name), identity);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TBAARootMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::llvm::StringRef identity) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getIdentityAttrName(odsState.name), odsBuilder.getStringAttr(identity));
}

void TBAARootMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::llvm::StringRef identity) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getIdentityAttrName(odsState.name), odsBuilder.getStringAttr(identity));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TBAARootMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TBAARootMetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_identity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'identity'");
    if (namedAttrIt->getName() == getIdentityAttrName()) {
      tblgen_identity = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_identity, "identity")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult TBAARootMetadataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TBAARootMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::StringAttr identityAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseLBrace())
    return ::mlir::failure();
  if (parser.parseKeyword("id"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(identityAttr, parser.getBuilder().getType<::mlir::NoneType>(), "identity",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRBrace())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TBAARootMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter << ' ';
  _odsPrinter << "{";
  _odsPrinter << "id";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIdentityAttr());
  _odsPrinter << "}";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  elidedAttrs.push_back("identity");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::TBAARootMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::TBAATagOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TBAATagOpGenericAdaptorBase::TBAATagOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.tbaa_tag", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TBAATagOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TBAATagOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr TBAATagOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, TBAATagOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef TBAATagOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr TBAATagOpGenericAdaptorBase::getBaseTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, TBAATagOp::getBaseTypeAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::StringRef TBAATagOpGenericAdaptorBase::getBaseType() {
  auto attr = getBaseTypeAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr TBAATagOpGenericAdaptorBase::getAccessTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, TBAATagOp::getAccessTypeAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::StringRef TBAATagOpGenericAdaptorBase::getAccessType() {
  auto attr = getAccessTypeAttr();
  return attr.getValue();
}

::mlir::IntegerAttr TBAATagOpGenericAdaptorBase::getOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, TBAATagOp::getOffsetAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t TBAATagOpGenericAdaptorBase::getOffset() {
  auto attr = getOffsetAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr TBAATagOpGenericAdaptorBase::getConstantAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 2, TBAATagOp::getConstantAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool TBAATagOpGenericAdaptorBase::getConstant() {
  auto attr = getConstantAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
TBAATagOpAdaptor::TBAATagOpAdaptor(TBAATagOp op) : TBAATagOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TBAATagOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_tag' op ""requires attribute 'access_type'");
    if (namedAttrIt->getName() == TBAATagOp::getAccessTypeAttrName(*odsOpName)) {
      tblgen_access_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_base_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_tag' op ""requires attribute 'base_type'");
    if (namedAttrIt->getName() == TBAATagOp::getBaseTypeAttrName(*odsOpName)) {
      tblgen_base_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_offset;
  ::mlir::Attribute tblgen_constant;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_tag' op ""requires attribute 'offset'");
    if (namedAttrIt->getName() == TBAATagOp::getOffsetAttrName(*odsOpName)) {
      tblgen_offset = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TBAATagOp::getConstantAttrName(*odsOpName)) {
      tblgen_constant = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_tag' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == TBAATagOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.tbaa_tag' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_base_type && !((tblgen_base_type.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.tbaa_tag' op ""attribute 'base_type' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_access_type && !((tblgen_access_type.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.tbaa_tag' op ""attribute 'access_type' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_offset && !(((tblgen_offset.isa<::mlir::IntegerAttr>())) && ((tblgen_offset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.tbaa_tag' op ""attribute 'offset' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_constant && !((tblgen_constant.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.tbaa_tag' op ""attribute 'constant' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TBAATagOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TBAATagOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TBAATagOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TBAATagOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr TBAATagOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef TBAATagOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr TBAATagOp::getBaseTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getBaseTypeAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef TBAATagOp::getBaseType() {
  auto attr = getBaseTypeAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr TBAATagOp::getAccessTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getAccessTypeAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef TBAATagOp::getAccessType() {
  auto attr = getAccessTypeAttr();
  return attr.getValue();
}

::mlir::IntegerAttr TBAATagOp::getOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getOffsetAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t TBAATagOp::getOffset() {
  auto attr = getOffsetAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr TBAATagOp::getConstantAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 2, getConstantAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TBAATagOp::getConstant() {
  auto attr = getConstantAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void TBAATagOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void TBAATagOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void TBAATagOp::setBaseTypeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getBaseTypeAttrName(), attr);
}

void TBAATagOp::setBaseType(::llvm::StringRef attrValue) {
  (*this)->setAttr(getBaseTypeAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void TBAATagOp::setAccessTypeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getAccessTypeAttrName(), attr);
}

void TBAATagOp::setAccessType(::llvm::StringRef attrValue) {
  (*this)->setAttr(getAccessTypeAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void TBAATagOp::setOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getOffsetAttrName(), attr);
}

void TBAATagOp::setOffset(uint64_t attrValue) {
  (*this)->setAttr(getOffsetAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void TBAATagOp::setConstantAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getConstantAttrName(), attr);
}

void TBAATagOp::setConstant(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getConstantAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getConstantAttrName());
}

::mlir::Attribute TBAATagOp::removeConstantAttr() {
  return (*this)->removeAttr(getConstantAttrName());
}

void TBAATagOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::FlatSymbolRefAttr base_type, ::mlir::FlatSymbolRefAttr access_type, ::mlir::IntegerAttr offset, /*optional*/::mlir::UnitAttr constant) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getBaseTypeAttrName(odsState.name), base_type);
  odsState.addAttribute(getAccessTypeAttrName(odsState.name), access_type);
  odsState.addAttribute(getOffsetAttrName(odsState.name), offset);
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
}

void TBAATagOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::FlatSymbolRefAttr base_type, ::mlir::FlatSymbolRefAttr access_type, ::mlir::IntegerAttr offset, /*optional*/::mlir::UnitAttr constant) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getBaseTypeAttrName(odsState.name), base_type);
  odsState.addAttribute(getAccessTypeAttrName(odsState.name), access_type);
  odsState.addAttribute(getOffsetAttrName(odsState.name), offset);
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TBAATagOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::llvm::StringRef base_type, ::llvm::StringRef access_type, uint64_t offset, /*optional*/bool constant) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getBaseTypeAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), base_type));
  odsState.addAttribute(getAccessTypeAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), access_type));
  odsState.addAttribute(getOffsetAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), offset));
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), ((constant) ? odsBuilder.getUnitAttr() : nullptr));
  }
}

void TBAATagOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::llvm::StringRef base_type, ::llvm::StringRef access_type, uint64_t offset, /*optional*/bool constant) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getBaseTypeAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), base_type));
  odsState.addAttribute(getAccessTypeAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), access_type));
  odsState.addAttribute(getOffsetAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), offset));
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), ((constant) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TBAATagOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TBAATagOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'access_type'");
    if (namedAttrIt->getName() == getAccessTypeAttrName()) {
      tblgen_access_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_base_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'base_type'");
    if (namedAttrIt->getName() == getBaseTypeAttrName()) {
      tblgen_base_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_offset;
  ::mlir::Attribute tblgen_constant;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'offset'");
    if (namedAttrIt->getName() == getOffsetAttrName()) {
      tblgen_offset = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getConstantAttrName()) {
      tblgen_constant = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_base_type, "base_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_access_type, "access_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_offset, "offset")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_constant, "constant")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult TBAATagOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TBAATagOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TBAATagOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getConstantAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("constant");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::TBAATagOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::TBAATypeDescriptorOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TBAATypeDescriptorOpGenericAdaptorBase::TBAATypeDescriptorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.tbaa_type_desc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TBAATypeDescriptorOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TBAATypeDescriptorOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr TBAATypeDescriptorOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, TBAATypeDescriptorOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef TBAATypeDescriptorOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr TBAATypeDescriptorOpGenericAdaptorBase::getIdentityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, TBAATypeDescriptorOp::getIdentityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > TBAATypeDescriptorOpGenericAdaptorBase::getIdentity() {
  auto attr = getIdentityAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr TBAATypeDescriptorOpGenericAdaptorBase::getMembersAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, TBAATypeDescriptorOp::getMembersAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TBAATypeDescriptorOpGenericAdaptorBase::getMembers() {
  auto attr = getMembersAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr TBAATypeDescriptorOpGenericAdaptorBase::getOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, TBAATypeDescriptorOp::getOffsetsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> TBAATypeDescriptorOpGenericAdaptorBase::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

} // namespace detail
TBAATypeDescriptorOpAdaptor::TBAATypeDescriptorOpAdaptor(TBAATypeDescriptorOp op) : TBAATypeDescriptorOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TBAATypeDescriptorOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_members;
  ::mlir::Attribute tblgen_identity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_type_desc' op ""requires attribute 'members'");
    if (namedAttrIt->getName() == TBAATypeDescriptorOp::getMembersAttrName(*odsOpName)) {
      tblgen_members = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TBAATypeDescriptorOp::getIdentityAttrName(*odsOpName)) {
      tblgen_identity = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_type_desc' op ""requires attribute 'offsets'");
    if (namedAttrIt->getName() == TBAATypeDescriptorOp::getOffsetsAttrName(*odsOpName)) {
      tblgen_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.tbaa_type_desc' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == TBAATypeDescriptorOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.tbaa_type_desc' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_identity && !((tblgen_identity.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.tbaa_type_desc' op ""attribute 'identity' failed to satisfy constraint: string attribute");

  if (tblgen_members && !(((tblgen_members.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_members.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.tbaa_type_desc' op ""attribute 'members' failed to satisfy constraint: flat symbol ref array attribute");

  if (tblgen_offsets && !((tblgen_offsets.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'llvm.tbaa_type_desc' op ""attribute 'offsets' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TBAATypeDescriptorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TBAATypeDescriptorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TBAATypeDescriptorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TBAATypeDescriptorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr TBAATypeDescriptorOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef TBAATypeDescriptorOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr TBAATypeDescriptorOp::getIdentityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getIdentityAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > TBAATypeDescriptorOp::getIdentity() {
  auto attr = getIdentityAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr TBAATypeDescriptorOp::getMembersAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getMembersAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TBAATypeDescriptorOp::getMembers() {
  auto attr = getMembersAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr TBAATypeDescriptorOp::getOffsetsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOffsetsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> TBAATypeDescriptorOp::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

void TBAATypeDescriptorOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void TBAATypeDescriptorOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void TBAATypeDescriptorOp::setIdentityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getIdentityAttrName(), attr);
}

void TBAATypeDescriptorOp::setIdentity(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIdentityAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getIdentityAttrName());
}

void TBAATypeDescriptorOp::setMembersAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMembersAttrName(), attr);
}

void TBAATypeDescriptorOp::setOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getOffsetsAttrName(), attr);
}

void TBAATypeDescriptorOp::setOffsets(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getOffsetsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

::mlir::Attribute TBAATypeDescriptorOp::removeIdentityAttr() {
  return (*this)->removeAttr(getIdentityAttrName());
}

void TBAATypeDescriptorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr identity, ::mlir::ArrayAttr members, ::mlir::DenseI64ArrayAttr offsets) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (identity) {
    odsState.addAttribute(getIdentityAttrName(odsState.name), identity);
  }
  odsState.addAttribute(getMembersAttrName(odsState.name), members);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
}

void TBAATypeDescriptorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr identity, ::mlir::ArrayAttr members, ::mlir::DenseI64ArrayAttr offsets) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (identity) {
    odsState.addAttribute(getIdentityAttrName(odsState.name), identity);
  }
  odsState.addAttribute(getMembersAttrName(odsState.name), members);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TBAATypeDescriptorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr identity, ::mlir::ArrayAttr members, ::llvm::ArrayRef<int64_t> offsets) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (identity) {
    odsState.addAttribute(getIdentityAttrName(odsState.name), identity);
  }
  odsState.addAttribute(getMembersAttrName(odsState.name), members);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(offsets));
}

void TBAATypeDescriptorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr identity, ::mlir::ArrayAttr members, ::llvm::ArrayRef<int64_t> offsets) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (identity) {
    odsState.addAttribute(getIdentityAttrName(odsState.name), identity);
  }
  odsState.addAttribute(getMembersAttrName(odsState.name), members);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(offsets));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TBAATypeDescriptorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TBAATypeDescriptorOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_members;
  ::mlir::Attribute tblgen_identity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'members'");
    if (namedAttrIt->getName() == getMembersAttrName()) {
      tblgen_members = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getIdentityAttrName()) {
      tblgen_identity = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'offsets'");
    if (namedAttrIt->getName() == getOffsetsAttrName()) {
      tblgen_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_identity, "identity")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps15(*this, tblgen_members, "members")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps12(*this, tblgen_offsets, "offsets")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult TBAATypeDescriptorOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TBAATypeDescriptorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::StringAttr identityAttr;
  ::mlir::ArrayAttr membersAttr;
  ::mlir::DenseI64ArrayAttr offsetsAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseLBrace())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("id"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(identityAttr, parser.getBuilder().getType<::mlir::NoneType>(), "identity",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseKeyword("members"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  if (parser.parseLBrace())
    return ::mlir::failure();
  {
    if (parseTBAAMembers(parser, membersAttr, offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("members", membersAttr);
    result.addAttribute("offsets", offsetsAttr);
  }
  if (parser.parseRBrace())
    return ::mlir::failure();
  if (parser.parseRBrace())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TBAATypeDescriptorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter << ' ';
  _odsPrinter << "{";
  if ((*this)->getAttr("identity")) {
    _odsPrinter << "id";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getIdentityAttr());
  }
  _odsPrinter << ",";
  _odsPrinter << ' ' << "members";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ' << "{";
  printTBAAMembers(_odsPrinter, *this, getMembersAttr(), getOffsetsAttr());
  _odsPrinter << "}";
  _odsPrinter << "}";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  elidedAttrs.push_back("identity");
  elidedAttrs.push_back("members");
  elidedAttrs.push_back("offsets");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::TBAATypeDescriptorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::TruncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TruncOpGenericAdaptorBase::TruncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.trunc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TruncOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TruncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TruncOpAdaptor::TruncOpAdaptor(TruncOp op) : TruncOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange TruncOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TruncOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TruncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TruncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TruncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::TruncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UDivOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UDivOpGenericAdaptorBase::UDivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.udiv", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UDivOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr UDivOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
UDivOpAdaptor::UDivOpAdaptor(UDivOp op) : UDivOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value UDivOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange UDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(UDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(UDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult UDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult UDivOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult UDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UIToFPOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UIToFPOpGenericAdaptorBase::UIToFPOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.uitofp", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UIToFPOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr UIToFPOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
UIToFPOpAdaptor::UIToFPOpAdaptor(UIToFPOp op) : UIToFPOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange UIToFPOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UIToFPOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UIToFPOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UIToFPOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UIToFPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UIToFPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::URemOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
URemOpGenericAdaptorBase::URemOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.urem", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> URemOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr URemOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
URemOpAdaptor::URemOpAdaptor(URemOp op) : URemOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult URemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> URemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range URemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value URemOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange URemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange URemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> URemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range URemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(URemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void URemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(URemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult URemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult URemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult URemOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult URemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void URemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void URemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::URemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UndefOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UndefOpGenericAdaptorBase::UndefOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.undef", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UndefOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr UndefOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
UndefOpAdaptor::UndefOpAdaptor(UndefOp op) : UndefOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UndefOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UndefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UndefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UndefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UndefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UndefOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UndefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UndefOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UndefOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UndefOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void UndefOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UndefOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UndefOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UnreachableOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UnreachableOpGenericAdaptorBase::UnreachableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.unreachable", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UnreachableOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr UnreachableOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
UnreachableOpAdaptor::UnreachableOpAdaptor(UnreachableOp op) : UnreachableOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UnreachableOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UnreachableOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnreachableOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UnreachableOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnreachableOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnreachableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void UnreachableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnreachableOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UnreachableOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult UnreachableOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UnreachableOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void UnreachableOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UnreachableOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::XOrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
XOrOpGenericAdaptorBase::XOrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.xor", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> XOrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr XOrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
XOrOpAdaptor::XOrOpAdaptor(XOrOp op) : XOrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult XOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> XOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range XOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::getLhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value XOrOp::getRhs() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange XOrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange XOrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> XOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range XOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(XOrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(XOrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult XOrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult XOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult XOrOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult XOrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void XOrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void XOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::XOrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ZExtOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ZExtOpGenericAdaptorBase::ZExtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.zext", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ZExtOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ZExtOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ZExtOpAdaptor::ZExtOpAdaptor(ZExtOp op) : ZExtOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ZExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ZExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ZExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOp::getArg() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ZExtOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ZExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ZExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOp::getRes() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ZExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ZExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ZExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ZExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ZExtOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ZExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ZExtOp)


#endif  // GET_OP_CLASSES

