/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue val) {
  switch (val) {
    case ClauseDefaultValue::Present: return "present";
    case ClauseDefaultValue::None: return "none";
  }
  return "";
}

::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseDefaultValue>>(str)
      .Case("present", ClauseDefaultValue::Present)
      .Case("none", ClauseDefaultValue::None)
      .Default(::std::nullopt);
}
::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(uint32_t value) {
  switch (value) {
  case 0: return ClauseDefaultValue::Present;
  case 1: return ClauseDefaultValue::None;
  default: return ::std::nullopt;
  }
}

} // namespace acc
} // namespace mlir

namespace mlir {
namespace acc {
::llvm::StringRef stringifyDataClause(DataClause val) {
  switch (val) {
    case DataClause::acc_copyin: return "acc_copyin";
    case DataClause::acc_copyin_readonly: return "acc_copyin_readonly";
    case DataClause::acc_copy: return "acc_copy";
    case DataClause::acc_copyout: return "acc_copyout";
    case DataClause::acc_copyout_zero: return "acc_copyout_zero";
    case DataClause::acc_present: return "acc_present";
    case DataClause::acc_create: return "acc_create";
    case DataClause::acc_create_zero: return "acc_create_zero";
    case DataClause::acc_delete: return "acc_delete";
    case DataClause::acc_attach: return "acc_attach";
    case DataClause::acc_detach: return "acc_detach";
    case DataClause::acc_no_create: return "acc_no_create";
    case DataClause::acc_private: return "acc_private";
    case DataClause::acc_firstprivate: return "acc_firstprivate";
    case DataClause::acc_deviceptr: return "acc_deviceptr";
    case DataClause::acc_getdeviceptr: return "acc_getdeviceptr";
  }
  return "";
}

::std::optional<DataClause> symbolizeDataClause(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DataClause>>(str)
      .Case("acc_copyin", DataClause::acc_copyin)
      .Case("acc_copyin_readonly", DataClause::acc_copyin_readonly)
      .Case("acc_copy", DataClause::acc_copy)
      .Case("acc_copyout", DataClause::acc_copyout)
      .Case("acc_copyout_zero", DataClause::acc_copyout_zero)
      .Case("acc_present", DataClause::acc_present)
      .Case("acc_create", DataClause::acc_create)
      .Case("acc_create_zero", DataClause::acc_create_zero)
      .Case("acc_delete", DataClause::acc_delete)
      .Case("acc_attach", DataClause::acc_attach)
      .Case("acc_detach", DataClause::acc_detach)
      .Case("acc_no_create", DataClause::acc_no_create)
      .Case("acc_private", DataClause::acc_private)
      .Case("acc_firstprivate", DataClause::acc_firstprivate)
      .Case("acc_deviceptr", DataClause::acc_deviceptr)
      .Case("acc_getdeviceptr", DataClause::acc_getdeviceptr)
      .Default(::std::nullopt);
}
::std::optional<DataClause> symbolizeDataClause(uint64_t value) {
  switch (value) {
  case 1: return DataClause::acc_copyin;
  case 2: return DataClause::acc_copyin_readonly;
  case 3: return DataClause::acc_copy;
  case 4: return DataClause::acc_copyout;
  case 5: return DataClause::acc_copyout_zero;
  case 6: return DataClause::acc_present;
  case 7: return DataClause::acc_create;
  case 8: return DataClause::acc_create_zero;
  case 9: return DataClause::acc_delete;
  case 10: return DataClause::acc_attach;
  case 11: return DataClause::acc_detach;
  case 12: return DataClause::acc_no_create;
  case 13: return DataClause::acc_private;
  case 14: return DataClause::acc_firstprivate;
  case 15: return DataClause::acc_deviceptr;
  case 16: return DataClause::acc_getdeviceptr;
  default: return ::std::nullopt;
  }
}

bool DataClauseAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)));
}
DataClauseAttr DataClauseAttr::get(::mlir::MLIRContext *context, DataClause val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<DataClauseAttr>();
}
DataClause DataClauseAttr::getValue() const {
  return static_cast<DataClause>(::mlir::IntegerAttr::getInt());
}
} // namespace acc
} // namespace mlir

namespace mlir {
namespace acc {
::llvm::StringRef stringifyReductionOp(ReductionOp val) {
  switch (val) {
    case ReductionOp::redop_add: return "redop_add";
    case ReductionOp::redop_mul: return "redop_mul";
    case ReductionOp::redop_max: return "redop_max";
    case ReductionOp::redop_min: return "redop_min";
    case ReductionOp::redop_and: return "redop_and";
    case ReductionOp::redop_or: return "redop_or";
    case ReductionOp::redop_xor: return "redop_xor";
    case ReductionOp::redop_leqv: return "redop_leqv";
    case ReductionOp::redop_lneqv: return "redop_lneqv";
    case ReductionOp::redop_land: return "redop_land";
    case ReductionOp::redop_lor: return "redop_lor";
  }
  return "";
}

::std::optional<ReductionOp> symbolizeReductionOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ReductionOp>>(str)
      .Case("redop_add", ReductionOp::redop_add)
      .Case("redop_mul", ReductionOp::redop_mul)
      .Case("redop_max", ReductionOp::redop_max)
      .Case("redop_min", ReductionOp::redop_min)
      .Case("redop_and", ReductionOp::redop_and)
      .Case("redop_or", ReductionOp::redop_or)
      .Case("redop_xor", ReductionOp::redop_xor)
      .Case("redop_leqv", ReductionOp::redop_leqv)
      .Case("redop_lneqv", ReductionOp::redop_lneqv)
      .Case("redop_land", ReductionOp::redop_land)
      .Case("redop_lor", ReductionOp::redop_lor)
      .Default(::std::nullopt);
}
::std::optional<ReductionOp> symbolizeReductionOp(uint32_t value) {
  switch (value) {
  case 0: return ReductionOp::redop_add;
  case 1: return ReductionOp::redop_mul;
  case 2: return ReductionOp::redop_max;
  case 3: return ReductionOp::redop_min;
  case 4: return ReductionOp::redop_and;
  case 5: return ReductionOp::redop_or;
  case 6: return ReductionOp::redop_xor;
  case 7: return ReductionOp::redop_leqv;
  case 8: return ReductionOp::redop_lneqv;
  case 9: return ReductionOp::redop_land;
  case 10: return ReductionOp::redop_lor;
  default: return ::std::nullopt;
  }
}

} // namespace acc
} // namespace mlir

