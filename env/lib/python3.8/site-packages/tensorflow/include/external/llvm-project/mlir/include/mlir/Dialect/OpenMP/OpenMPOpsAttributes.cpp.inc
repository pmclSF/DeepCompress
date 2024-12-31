/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::omp::ClauseCancellationConstructTypeAttr,
::mlir::omp::ClauseDependAttr,
::mlir::omp::ClauseTaskDependAttr,
::mlir::omp::FlagsAttr,
::mlir::omp::ClauseGrainsizeTypeAttr,
::mlir::omp::IsDeviceAttr,
::mlir::omp::ClauseMemoryOrderKindAttr,
::mlir::omp::ClauseNumTasksTypeAttr,
::mlir::omp::ClauseOrderKindAttr,
::mlir::omp::ClauseProcBindKindAttr,
::mlir::omp::ClauseScheduleKindAttr,
::mlir::omp::ScheduleModifierAttr,
::mlir::omp::TargetAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::omp::ClauseCancellationConstructTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseCancellationConstructTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseDependAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseDependAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseTaskDependAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseTaskDependAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::FlagsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::FlagsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseGrainsizeTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseGrainsizeTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::IsDeviceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::IsDeviceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseMemoryOrderKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseMemoryOrderKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseNumTasksTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseNumTasksTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseOrderKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseOrderKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseProcBindKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseProcBindKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseScheduleKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseScheduleKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ScheduleModifierAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ScheduleModifierAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::TargetAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::TargetAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::omp::ClauseCancellationConstructTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseCancellationConstructTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseDependAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseDependAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseTaskDependAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseTaskDependAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::FlagsAttr>([&](auto t) {
      printer << ::mlir::omp::FlagsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseGrainsizeTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseGrainsizeTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::IsDeviceAttr>([&](auto t) {
      printer << ::mlir::omp::IsDeviceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseMemoryOrderKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseMemoryOrderKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseNumTasksTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseNumTasksTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseOrderKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseOrderKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseProcBindKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseProcBindKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseScheduleKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseScheduleKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ScheduleModifierAttr>([&](auto t) {
      printer << ::mlir::omp::ScheduleModifierAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::TargetAttr>([&](auto t) {
      printer << ::mlir::omp::TargetAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace omp {
namespace detail {
struct ClauseCancellationConstructTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseCancellationConstructType>;
  ClauseCancellationConstructTypeAttrStorage(::mlir::omp::ClauseCancellationConstructType value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseCancellationConstructTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseCancellationConstructTypeAttrStorage>()) ClauseCancellationConstructTypeAttrStorage(value);
  }

  ::mlir::omp::ClauseCancellationConstructType value;
};
} // namespace detail
ClauseCancellationConstructTypeAttr ClauseCancellationConstructTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseCancellationConstructType value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseCancellationConstructTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseCancellationConstructType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseCancellationConstructType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseCancellationConstructType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseCancellationConstructType" << " to be one of: " << "parallel" << ", " << "loop" << ", " << "sections" << ", " << "taskgroup")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CancellationConstructTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseCancellationConstructType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseCancellationConstructTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseCancellationConstructType((*_result_value)));
}

void ClauseCancellationConstructTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseCancellationConstructType(getValue());
}

::mlir::omp::ClauseCancellationConstructType ClauseCancellationConstructTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseCancellationConstructTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseDependAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseDepend>;
  ClauseDependAttrStorage(::mlir::omp::ClauseDepend value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseDependAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseDependAttrStorage>()) ClauseDependAttrStorage(value);
  }

  ::mlir::omp::ClauseDepend value;
};
} // namespace detail
ClauseDependAttr ClauseDependAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseDepend value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseDependAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseDepend> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseDepend> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseDepend(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseDepend" << " to be one of: " << "dependsource" << ", " << "dependsink")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ClauseDependAttr parameter 'value' which is to be a `::mlir::omp::ClauseDepend`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return ClauseDependAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseDepend((*_result_value)));
}

void ClauseDependAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyClauseDepend(getValue());
  odsPrinter << ")";
}

::mlir::omp::ClauseDepend ClauseDependAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseDependAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseTaskDependAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseTaskDepend>;
  ClauseTaskDependAttrStorage(::mlir::omp::ClauseTaskDepend value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseTaskDependAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseTaskDependAttrStorage>()) ClauseTaskDependAttrStorage(value);
  }

  ::mlir::omp::ClauseTaskDepend value;
};
} // namespace detail
ClauseTaskDependAttr ClauseTaskDependAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseTaskDepend value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseTaskDependAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseTaskDepend> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseTaskDepend> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseTaskDepend(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseTaskDepend" << " to be one of: " << "taskdependin" << ", " << "taskdependout" << ", " << "taskdependinout")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ClauseTaskDependAttr parameter 'value' which is to be a `::mlir::omp::ClauseTaskDepend`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return ClauseTaskDependAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseTaskDepend((*_result_value)));
}

void ClauseTaskDependAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyClauseTaskDepend(getValue());
  odsPrinter << ")";
}

::mlir::omp::ClauseTaskDepend ClauseTaskDependAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseTaskDependAttr)
namespace mlir {
namespace omp {
namespace detail {
struct FlagsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<uint32_t, bool, bool, bool, bool>;
  FlagsAttrStorage(uint32_t debug_kind, bool assume_teams_oversubscription, bool assume_threads_oversubscription, bool assume_no_thread_state, bool assume_no_nested_parallelism) : debug_kind(debug_kind), assume_teams_oversubscription(assume_teams_oversubscription), assume_threads_oversubscription(assume_threads_oversubscription), assume_no_thread_state(assume_no_thread_state), assume_no_nested_parallelism(assume_no_nested_parallelism) {}

  KeyTy getAsKey() const {
    return KeyTy(debug_kind, assume_teams_oversubscription, assume_threads_oversubscription, assume_no_thread_state, assume_no_nested_parallelism);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (debug_kind == std::get<0>(tblgenKey)) && (assume_teams_oversubscription == std::get<1>(tblgenKey)) && (assume_threads_oversubscription == std::get<2>(tblgenKey)) && (assume_no_thread_state == std::get<3>(tblgenKey)) && (assume_no_nested_parallelism == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static FlagsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto debug_kind = std::get<0>(tblgenKey);
    auto assume_teams_oversubscription = std::get<1>(tblgenKey);
    auto assume_threads_oversubscription = std::get<2>(tblgenKey);
    auto assume_no_thread_state = std::get<3>(tblgenKey);
    auto assume_no_nested_parallelism = std::get<4>(tblgenKey);
    return new (allocator.allocate<FlagsAttrStorage>()) FlagsAttrStorage(debug_kind, assume_teams_oversubscription, assume_threads_oversubscription, assume_no_thread_state, assume_no_nested_parallelism);
  }

  uint32_t debug_kind;
  bool assume_teams_oversubscription;
  bool assume_threads_oversubscription;
  bool assume_no_thread_state;
  bool assume_no_nested_parallelism;
};
} // namespace detail
FlagsAttr FlagsAttr::get(::mlir::MLIRContext *context, uint32_t debug_kind, bool assume_teams_oversubscription, bool assume_threads_oversubscription, bool assume_no_thread_state, bool assume_no_nested_parallelism) {
  return Base::get(context, debug_kind, assume_teams_oversubscription, assume_threads_oversubscription, assume_no_thread_state, assume_no_nested_parallelism);
}

::mlir::Attribute FlagsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<uint32_t> _result_debug_kind;
  ::mlir::FailureOr<bool> _result_assume_teams_oversubscription;
  ::mlir::FailureOr<bool> _result_assume_threads_oversubscription;
  ::mlir::FailureOr<bool> _result_assume_no_thread_state;
  ::mlir::FailureOr<bool> _result_assume_no_nested_parallelism;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_debug_kind = false;
  bool _seen_assume_teams_oversubscription = false;
  bool _seen_assume_threads_oversubscription = false;
  bool _seen_assume_no_thread_state = false;
  bool _seen_assume_no_nested_parallelism = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_debug_kind && _paramKey == "debug_kind") {
        _seen_debug_kind = true;

        // Parse variable 'debug_kind'
        _result_debug_kind = ::mlir::FieldParser<uint32_t>::parse(odsParser);
        if (::mlir::failed(_result_debug_kind)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'debug_kind' which is to be a `uint32_t`");
          return {};
        }
      } else if (!_seen_assume_teams_oversubscription && _paramKey == "assume_teams_oversubscription") {
        _seen_assume_teams_oversubscription = true;

        // Parse variable 'assume_teams_oversubscription'
        _result_assume_teams_oversubscription = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_teams_oversubscription)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_teams_oversubscription' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_assume_threads_oversubscription && _paramKey == "assume_threads_oversubscription") {
        _seen_assume_threads_oversubscription = true;

        // Parse variable 'assume_threads_oversubscription'
        _result_assume_threads_oversubscription = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_threads_oversubscription)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_threads_oversubscription' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_assume_no_thread_state && _paramKey == "assume_no_thread_state") {
        _seen_assume_no_thread_state = true;

        // Parse variable 'assume_no_thread_state'
        _result_assume_no_thread_state = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_no_thread_state)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_no_thread_state' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_assume_no_nested_parallelism && _paramKey == "assume_no_nested_parallelism") {
        _seen_assume_no_nested_parallelism = true;

        // Parse variable 'assume_no_nested_parallelism'
        _result_assume_no_nested_parallelism = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_no_nested_parallelism)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_no_nested_parallelism' which is to be a `bool`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return FlagsAttr::get(odsParser.getContext(),
      uint32_t((_result_debug_kind.value_or(0))),
      bool((_result_assume_teams_oversubscription.value_or(false))),
      bool((_result_assume_threads_oversubscription.value_or(false))),
      bool((_result_assume_no_thread_state.value_or(false))),
      bool((_result_assume_no_nested_parallelism.value_or(false))));
}

void FlagsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getDebugKind() == 0)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "debug_kind = ";
      if (!(getDebugKind() == 0)) {
        odsPrinter.printStrippedAttrOrType(getDebugKind());
      }
    }
    if (!(getAssumeTeamsOversubscription() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_teams_oversubscription = ";
      if (!(getAssumeTeamsOversubscription() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeTeamsOversubscription());
      }
    }
    if (!(getAssumeThreadsOversubscription() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_threads_oversubscription = ";
      if (!(getAssumeThreadsOversubscription() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeThreadsOversubscription());
      }
    }
    if (!(getAssumeNoThreadState() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_no_thread_state = ";
      if (!(getAssumeNoThreadState() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeNoThreadState());
      }
    }
    if (!(getAssumeNoNestedParallelism() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_no_nested_parallelism = ";
      if (!(getAssumeNoNestedParallelism() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeNoNestedParallelism());
      }
    }
  }
  odsPrinter << ">";
}

uint32_t FlagsAttr::getDebugKind() const {
  return getImpl()->debug_kind;
}

bool FlagsAttr::getAssumeTeamsOversubscription() const {
  return getImpl()->assume_teams_oversubscription;
}

bool FlagsAttr::getAssumeThreadsOversubscription() const {
  return getImpl()->assume_threads_oversubscription;
}

bool FlagsAttr::getAssumeNoThreadState() const {
  return getImpl()->assume_no_thread_state;
}

bool FlagsAttr::getAssumeNoNestedParallelism() const {
  return getImpl()->assume_no_nested_parallelism;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::FlagsAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseGrainsizeTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseGrainsizeType>;
  ClauseGrainsizeTypeAttrStorage(::mlir::omp::ClauseGrainsizeType value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseGrainsizeTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseGrainsizeTypeAttrStorage>()) ClauseGrainsizeTypeAttrStorage(value);
  }

  ::mlir::omp::ClauseGrainsizeType value;
};
} // namespace detail
ClauseGrainsizeTypeAttr ClauseGrainsizeTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseGrainsizeType value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseGrainsizeTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseGrainsizeType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseGrainsizeType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseGrainsizeType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseGrainsizeType" << " to be one of: " << "strict")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GrainsizeTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseGrainsizeType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseGrainsizeTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseGrainsizeType((*_result_value)));
}

void ClauseGrainsizeTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseGrainsizeType(getValue());
}

::mlir::omp::ClauseGrainsizeType ClauseGrainsizeTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseGrainsizeTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct IsDeviceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<bool>;
  IsDeviceAttrStorage(bool is_device) : is_device(is_device) {}

  KeyTy getAsKey() const {
    return KeyTy(is_device);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (is_device == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IsDeviceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto is_device = std::get<0>(tblgenKey);
    return new (allocator.allocate<IsDeviceAttrStorage>()) IsDeviceAttrStorage(is_device);
  }

  bool is_device;
};
} // namespace detail
IsDeviceAttr IsDeviceAttr::get(::mlir::MLIRContext *context, bool is_device) {
  return Base::get(context, is_device);
}

::mlir::Attribute IsDeviceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<bool> _result_is_device;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_is_device = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_is_device && _paramKey == "is_device") {
        _seen_is_device = true;

        // Parse variable 'is_device'
        _result_is_device = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_is_device)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse IsDeviceAttr parameter 'is_device' which is to be a `bool`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_is_device));
  return IsDeviceAttr::get(odsParser.getContext(),
      bool((*_result_is_device)));
}

void IsDeviceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "is_device = ";
    odsPrinter.printStrippedAttrOrType(getIsDevice());
  }
  odsPrinter << ">";
}

bool IsDeviceAttr::getIsDevice() const {
  return getImpl()->is_device;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::IsDeviceAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseMemoryOrderKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseMemoryOrderKind>;
  ClauseMemoryOrderKindAttrStorage(::mlir::omp::ClauseMemoryOrderKind value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseMemoryOrderKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseMemoryOrderKindAttrStorage>()) ClauseMemoryOrderKindAttrStorage(value);
  }

  ::mlir::omp::ClauseMemoryOrderKind value;
};
} // namespace detail
ClauseMemoryOrderKindAttr ClauseMemoryOrderKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseMemoryOrderKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseMemoryOrderKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseMemoryOrderKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseMemoryOrderKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseMemoryOrderKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseMemoryOrderKind" << " to be one of: " << "seq_cst" << ", " << "acq_rel" << ", " << "acquire" << ", " << "release" << ", " << "relaxed")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MemoryOrderKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseMemoryOrderKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseMemoryOrderKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseMemoryOrderKind((*_result_value)));
}

void ClauseMemoryOrderKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseMemoryOrderKind(getValue());
}

::mlir::omp::ClauseMemoryOrderKind ClauseMemoryOrderKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseMemoryOrderKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseNumTasksTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseNumTasksType>;
  ClauseNumTasksTypeAttrStorage(::mlir::omp::ClauseNumTasksType value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseNumTasksTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseNumTasksTypeAttrStorage>()) ClauseNumTasksTypeAttrStorage(value);
  }

  ::mlir::omp::ClauseNumTasksType value;
};
} // namespace detail
ClauseNumTasksTypeAttr ClauseNumTasksTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseNumTasksType value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseNumTasksTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseNumTasksType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseNumTasksType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseNumTasksType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseNumTasksType" << " to be one of: " << "strict")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NumTasksTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseNumTasksType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseNumTasksTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseNumTasksType((*_result_value)));
}

void ClauseNumTasksTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseNumTasksType(getValue());
}

::mlir::omp::ClauseNumTasksType ClauseNumTasksTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseNumTasksTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseOrderKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseOrderKind>;
  ClauseOrderKindAttrStorage(::mlir::omp::ClauseOrderKind value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseOrderKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseOrderKindAttrStorage>()) ClauseOrderKindAttrStorage(value);
  }

  ::mlir::omp::ClauseOrderKind value;
};
} // namespace detail
ClauseOrderKindAttr ClauseOrderKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseOrderKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseOrderKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseOrderKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseOrderKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseOrderKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseOrderKind" << " to be one of: " << "concurrent")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OrderKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseOrderKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseOrderKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseOrderKind((*_result_value)));
}

void ClauseOrderKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseOrderKind(getValue());
}

::mlir::omp::ClauseOrderKind ClauseOrderKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseOrderKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseProcBindKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseProcBindKind>;
  ClauseProcBindKindAttrStorage(::mlir::omp::ClauseProcBindKind value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseProcBindKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseProcBindKindAttrStorage>()) ClauseProcBindKindAttrStorage(value);
  }

  ::mlir::omp::ClauseProcBindKind value;
};
} // namespace detail
ClauseProcBindKindAttr ClauseProcBindKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseProcBindKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseProcBindKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseProcBindKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseProcBindKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseProcBindKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseProcBindKind" << " to be one of: " << "primary" << ", " << "master" << ", " << "close" << ", " << "spread")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ProcBindKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseProcBindKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseProcBindKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseProcBindKind((*_result_value)));
}

void ClauseProcBindKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseProcBindKind(getValue());
}

::mlir::omp::ClauseProcBindKind ClauseProcBindKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseProcBindKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseScheduleKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseScheduleKind>;
  ClauseScheduleKindAttrStorage(::mlir::omp::ClauseScheduleKind value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseScheduleKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseScheduleKindAttrStorage>()) ClauseScheduleKindAttrStorage(value);
  }

  ::mlir::omp::ClauseScheduleKind value;
};
} // namespace detail
ClauseScheduleKindAttr ClauseScheduleKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseScheduleKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseScheduleKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseScheduleKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseScheduleKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseScheduleKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseScheduleKind" << " to be one of: " << "static" << ", " << "dynamic" << ", " << "guided" << ", " << "auto" << ", " << "runtime")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScheduleKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseScheduleKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseScheduleKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseScheduleKind((*_result_value)));
}

void ClauseScheduleKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseScheduleKind(getValue());
}

::mlir::omp::ClauseScheduleKind ClauseScheduleKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseScheduleKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ScheduleModifierAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ScheduleModifier>;
  ScheduleModifierAttrStorage(::mlir::omp::ScheduleModifier value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ScheduleModifierAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ScheduleModifierAttrStorage>()) ScheduleModifierAttrStorage(value);
  }

  ::mlir::omp::ScheduleModifier value;
};
} // namespace detail
ScheduleModifierAttr ScheduleModifierAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ScheduleModifier value) {
  return Base::get(context, value);
}

::mlir::Attribute ScheduleModifierAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ScheduleModifier> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ScheduleModifier> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeScheduleModifier(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ScheduleModifier" << " to be one of: " << "none" << ", " << "monotonic" << ", " << "nonmonotonic" << ", " << "simd")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScheduleModifierAttr parameter 'value' which is to be a `::mlir::omp::ScheduleModifier`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ScheduleModifierAttr::get(odsParser.getContext(),
      ::mlir::omp::ScheduleModifier((*_result_value)));
}

void ScheduleModifierAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyScheduleModifier(getValue());
}

::mlir::omp::ScheduleModifier ScheduleModifierAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ScheduleModifierAttr)
namespace mlir {
namespace omp {
namespace detail {
struct TargetAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::StringRef, ::llvm::StringRef>;
  TargetAttrStorage(::llvm::StringRef target_cpu, ::llvm::StringRef target_features) : target_cpu(target_cpu), target_features(target_features) {}

  KeyTy getAsKey() const {
    return KeyTy(target_cpu, target_features);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (target_cpu == std::get<0>(tblgenKey)) && (target_features == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TargetAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto target_cpu = std::get<0>(tblgenKey);
    auto target_features = std::get<1>(tblgenKey);
    target_cpu = allocator.copyInto(target_cpu);
    target_features = allocator.copyInto(target_features);
    return new (allocator.allocate<TargetAttrStorage>()) TargetAttrStorage(target_cpu, target_features);
  }

  ::llvm::StringRef target_cpu;
  ::llvm::StringRef target_features;
};
} // namespace detail
TargetAttr TargetAttr::get(::mlir::MLIRContext *context, ::llvm::StringRef target_cpu, ::llvm::StringRef target_features) {
  return Base::get(context, target_cpu, target_features);
}

::mlir::Attribute TargetAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::string> _result_target_cpu;
  ::mlir::FailureOr<std::string> _result_target_features;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_target_cpu = false;
  bool _seen_target_features = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_target_cpu && _paramKey == "target_cpu") {
        _seen_target_cpu = true;

        // Parse variable 'target_cpu'
        _result_target_cpu = ::mlir::FieldParser<std::string>::parse(odsParser);
        if (::mlir::failed(_result_target_cpu)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TargetAttr parameter 'target_cpu' which is to be a `::llvm::StringRef`");
          return {};
        }
      } else if (!_seen_target_features && _paramKey == "target_features") {
        _seen_target_features = true;

        // Parse variable 'target_features'
        _result_target_features = ::mlir::FieldParser<std::string>::parse(odsParser);
        if (::mlir::failed(_result_target_features)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TargetAttr parameter 'target_features' which is to be a `::llvm::StringRef`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_target_cpu));
  assert(::mlir::succeeded(_result_target_features));
  return TargetAttr::get(odsParser.getContext(),
      ::llvm::StringRef((*_result_target_cpu)),
      ::llvm::StringRef((*_result_target_features)));
}

void TargetAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "target_cpu = ";
    odsPrinter << '"' << getTargetCpu() << '"';;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "target_features = ";
    odsPrinter << '"' << getTargetFeatures() << '"';;
  }
  odsPrinter << ">";
}

::llvm::StringRef TargetAttr::getTargetCpu() const {
  return getImpl()->target_cpu;
}

::llvm::StringRef TargetAttr::getTargetFeatures() const {
  return getImpl()->target_features;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TargetAttr)
namespace mlir {
namespace omp {

/// Parse an attribute registered to this dialect.
::mlir::Attribute OpenMPDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void OpenMPDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace omp
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

