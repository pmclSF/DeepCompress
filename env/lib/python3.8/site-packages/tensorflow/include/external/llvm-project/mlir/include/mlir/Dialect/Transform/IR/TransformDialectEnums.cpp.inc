/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
::llvm::StringRef stringifyFailurePropagationMode(FailurePropagationMode val) {
  switch (val) {
    case FailurePropagationMode::Propagate: return "propagate";
    case FailurePropagationMode::Suppress: return "suppress";
  }
  return "";
}

::std::optional<FailurePropagationMode> symbolizeFailurePropagationMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FailurePropagationMode>>(str)
      .Case("propagate", FailurePropagationMode::Propagate)
      .Case("suppress", FailurePropagationMode::Suppress)
      .Default(::std::nullopt);
}
::std::optional<FailurePropagationMode> symbolizeFailurePropagationMode(uint32_t value) {
  switch (value) {
  case 1: return FailurePropagationMode::Propagate;
  case 2: return FailurePropagationMode::Suppress;
  default: return ::std::nullopt;
  }
}

bool FailurePropagationModeAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
FailurePropagationModeAttr FailurePropagationModeAttr::get(::mlir::MLIRContext *context, FailurePropagationMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<FailurePropagationModeAttr>();
}
FailurePropagationMode FailurePropagationModeAttr::getValue() const {
  return static_cast<FailurePropagationMode>(::mlir::IntegerAttr::getInt());
}
} // namespace transform
} // namespace mlir

namespace mlir {
namespace transform {
::llvm::StringRef stringifyMatchCmpIPredicate(MatchCmpIPredicate val) {
  switch (val) {
    case MatchCmpIPredicate::eq: return "eq";
    case MatchCmpIPredicate::ne: return "ne";
    case MatchCmpIPredicate::lt: return "lt";
    case MatchCmpIPredicate::le: return "le";
    case MatchCmpIPredicate::gt: return "gt";
    case MatchCmpIPredicate::ge: return "ge";
  }
  return "";
}

::std::optional<MatchCmpIPredicate> symbolizeMatchCmpIPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MatchCmpIPredicate>>(str)
      .Case("eq", MatchCmpIPredicate::eq)
      .Case("ne", MatchCmpIPredicate::ne)
      .Case("lt", MatchCmpIPredicate::lt)
      .Case("le", MatchCmpIPredicate::le)
      .Case("gt", MatchCmpIPredicate::gt)
      .Case("ge", MatchCmpIPredicate::ge)
      .Default(::std::nullopt);
}
::std::optional<MatchCmpIPredicate> symbolizeMatchCmpIPredicate(uint32_t value) {
  switch (value) {
  case 0: return MatchCmpIPredicate::eq;
  case 1: return MatchCmpIPredicate::ne;
  case 2: return MatchCmpIPredicate::lt;
  case 3: return MatchCmpIPredicate::le;
  case 4: return MatchCmpIPredicate::gt;
  case 5: return MatchCmpIPredicate::ge;
  default: return ::std::nullopt;
  }
}

bool MatchCmpIPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)));
}
MatchCmpIPredicateAttr MatchCmpIPredicateAttr::get(::mlir::MLIRContext *context, MatchCmpIPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<MatchCmpIPredicateAttr>();
}
MatchCmpIPredicate MatchCmpIPredicateAttr::getValue() const {
  return static_cast<MatchCmpIPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace transform
} // namespace mlir

