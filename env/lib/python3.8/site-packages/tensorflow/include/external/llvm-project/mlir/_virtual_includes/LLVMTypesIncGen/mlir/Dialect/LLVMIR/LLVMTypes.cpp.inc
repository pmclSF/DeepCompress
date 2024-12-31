/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::LLVM::LLVMArrayType,
::mlir::LLVM::LLVMFixedVectorType,
::mlir::LLVM::LLVMFunctionType,
::mlir::LLVM::LLVMPointerType,
::mlir::LLVM::LLVMScalableVectorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::LLVM::LLVMArrayType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMArrayType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMFixedVectorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMFixedVectorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMFunctionType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMFunctionType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMPointerType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMPointerType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMScalableVectorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMScalableVectorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)    .Case<::mlir::LLVM::LLVMArrayType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMArrayType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMFixedVectorType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMFixedVectorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMFunctionType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMFunctionType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMPointerType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMPointerType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMScalableVectorType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMScalableVectorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMArrayTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  LLVMArrayTypeStorage(Type elementType, unsigned numElements) : elementType(elementType), numElements(numElements) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, numElements);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (numElements == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMArrayTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto numElements = std::get<1>(tblgenKey);
    return new (allocator.allocate<LLVMArrayTypeStorage>()) LLVMArrayTypeStorage(elementType, numElements);
  }

  Type elementType;
  unsigned numElements;
};
} // namespace detail
LLVMArrayType LLVMArrayType::get(::mlir::MLIRContext *context, Type elementType, unsigned numElements) {
  return Base::get(context, elementType, numElements);
}

LLVMArrayType LLVMArrayType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, unsigned numElements) {
  return Base::getChecked(emitError, context, elementType, numElements);
}

::mlir::Type LLVMArrayType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_numElements;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'numElements'
  _result_numElements = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_numElements)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMArrayType parameter 'numElements' which is to be a `unsigned`");
    return {};
  }
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_elementType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_elementType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'elementType'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_numElements));
  return odsParser.getChecked<LLVMArrayType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)),
      unsigned((*_result_numElements)));
}

void LLVMArrayType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getNumElements());
  odsPrinter << ' ' << "x";
  odsPrinter << ' ';
  printPrettyLLVMType(odsPrinter,
    getElementType());
  odsPrinter << ">";
}

Type LLVMArrayType::getElementType() const {
  return getImpl()->elementType;
}

unsigned LLVMArrayType::getNumElements() const {
  return getImpl()->numElements;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMArrayType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMFixedVectorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  LLVMFixedVectorTypeStorage(Type elementType, unsigned numElements) : elementType(elementType), numElements(numElements) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, numElements);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (numElements == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMFixedVectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto numElements = std::get<1>(tblgenKey);
    return new (allocator.allocate<LLVMFixedVectorTypeStorage>()) LLVMFixedVectorTypeStorage(elementType, numElements);
  }

  Type elementType;
  unsigned numElements;
};
} // namespace detail
LLVMFixedVectorType LLVMFixedVectorType::get(::mlir::MLIRContext *context, Type elementType, unsigned numElements) {
  return Base::get(context, elementType, numElements);
}

LLVMFixedVectorType LLVMFixedVectorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, unsigned numElements) {
  return Base::getChecked(emitError, context, elementType, numElements);
}

::mlir::Type LLVMFixedVectorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_numElements;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'numElements'
  _result_numElements = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_numElements)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMFixedVectorType parameter 'numElements' which is to be a `unsigned`");
    return {};
  }
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_elementType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_elementType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'elementType'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_numElements));
  return odsParser.getChecked<LLVMFixedVectorType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)),
      unsigned((*_result_numElements)));
}

void LLVMFixedVectorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getNumElements());
  odsPrinter << ' ' << "x";
  odsPrinter << ' ';
  printPrettyLLVMType(odsPrinter,
    getElementType());
  odsPrinter << ">";
}

Type LLVMFixedVectorType::getElementType() const {
  return getImpl()->elementType;
}

unsigned LLVMFixedVectorType::getNumElements() const {
  return getImpl()->numElements;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMFixedVectorType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMFunctionTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, ::llvm::ArrayRef<Type>, bool>;
  LLVMFunctionTypeStorage(Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) : returnType(returnType), params(params), varArg(varArg) {}

  KeyTy getAsKey() const {
    return KeyTy(returnType, params, varArg);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (returnType == std::get<0>(tblgenKey)) && (params == std::get<1>(tblgenKey)) && (varArg == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static LLVMFunctionTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto returnType = std::get<0>(tblgenKey);
    auto params = std::get<1>(tblgenKey);
    auto varArg = std::get<2>(tblgenKey);
    params = allocator.copyInto(params);
    return new (allocator.allocate<LLVMFunctionTypeStorage>()) LLVMFunctionTypeStorage(returnType, params, varArg);
  }

  Type returnType;
  ::llvm::ArrayRef<Type> params;
  bool varArg;
};
} // namespace detail
LLVMFunctionType LLVMFunctionType::get(::mlir::MLIRContext *context, Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) {
  return Base::get(context, returnType, params, varArg);
}

LLVMFunctionType LLVMFunctionType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) {
  return Base::getChecked(emitError, context, returnType, params, varArg);
}

::mlir::Type LLVMFunctionType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_returnType;
  ::mlir::FailureOr<::llvm::SmallVector<Type>> _result_params;
  ::mlir::FailureOr<bool> _result_varArg;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_returnType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_returnType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'returnType'");
      return {};
    }
  }
  // Parse literal '('
  if (odsParser.parseLParen()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseFunctionTypes(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_params),
      ::mlir::detail::unwrapForCustomParse(_result_varArg));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_params)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'params'");
      return {};
    }
    if (::mlir::failed(_result_varArg)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'varArg'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_returnType));
  assert(::mlir::succeeded(_result_params));
  assert(::mlir::succeeded(_result_varArg));
  return odsParser.getChecked<LLVMFunctionType>(odsLoc, odsParser.getContext(),
      Type((*_result_returnType)),
      ::llvm::ArrayRef<Type>((*_result_params)),
      bool((*_result_varArg)));
}

void LLVMFunctionType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyLLVMType(odsPrinter,
    getReturnType());
  odsPrinter << " ";
  odsPrinter << "(";
  printFunctionTypes(odsPrinter,
    getParams(),
    getVarArg());
  odsPrinter << ">";
}

Type LLVMFunctionType::getReturnType() const {
  return getImpl()->returnType;
}

::llvm::ArrayRef<Type> LLVMFunctionType::getParams() const {
  return getImpl()->params;
}

bool LLVMFunctionType::getVarArg() const {
  return getImpl()->varArg;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMFunctionType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMPointerTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  LLVMPointerTypeStorage(Type elementType, unsigned addressSpace) : elementType(elementType), addressSpace(addressSpace) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, addressSpace);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (addressSpace == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMPointerTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto addressSpace = std::get<1>(tblgenKey);
    return new (allocator.allocate<LLVMPointerTypeStorage>()) LLVMPointerTypeStorage(elementType, addressSpace);
  }

  Type elementType;
  unsigned addressSpace;
};
} // namespace detail
LLVMPointerType LLVMPointerType::get(::mlir::MLIRContext *context, Type elementType, unsigned addressSpace) {
  return Base::get(context, elementType, addressSpace);
}

LLVMPointerType LLVMPointerType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, unsigned addressSpace) {
  return Base::getChecked(emitError, context, elementType, addressSpace);
}

LLVMPointerType LLVMPointerType::get(::mlir::MLIRContext *context, unsigned addressSpace) {
  return Base::get(context, Type(), addressSpace);
}

LLVMPointerType LLVMPointerType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned addressSpace) {
  return Base::getChecked(emitError, context, Type(), addressSpace);
}

::mlir::Type LLVMPointerType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_addressSpace;
  // Parse literal '<'
  if (odsParser.parseOptionalLess()) {
  } else {
    {
      auto odsCustomLoc = odsParser.getCurrentLocation();
      (void)odsCustomLoc;
      auto odsCustomResult = parsePointer(odsParser,
        ::mlir::detail::unwrapForCustomParse(_result_elementType),
        ::mlir::detail::unwrapForCustomParse(_result_addressSpace));
      if (::mlir::failed(odsCustomResult)) return {};
    }
    // Parse literal '>'
    if (odsParser.parseGreater()) return {};
  }
  return odsParser.getChecked<LLVMPointerType>(odsLoc, odsParser.getContext(),
      Type((_result_elementType.value_or(Type()))),
      unsigned((_result_addressSpace.value_or(0))));
}

void LLVMPointerType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  if (!(getElementType() == Type()) || !(getAddressSpace() == 0)) {
    odsPrinter << "<";
    printPointer(odsPrinter,
      getElementType(),
      getAddressSpace());
    odsPrinter << ">";
  } else {
  }
}

Type LLVMPointerType::getElementType() const {
  return getImpl()->elementType;
}

unsigned LLVMPointerType::getAddressSpace() const {
  return getImpl()->addressSpace;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMPointerType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMScalableVectorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  LLVMScalableVectorTypeStorage(Type elementType, unsigned minNumElements) : elementType(elementType), minNumElements(minNumElements) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, minNumElements);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (minNumElements == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMScalableVectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto minNumElements = std::get<1>(tblgenKey);
    return new (allocator.allocate<LLVMScalableVectorTypeStorage>()) LLVMScalableVectorTypeStorage(elementType, minNumElements);
  }

  Type elementType;
  unsigned minNumElements;
};
} // namespace detail
LLVMScalableVectorType LLVMScalableVectorType::get(::mlir::MLIRContext *context, Type elementType, unsigned minNumElements) {
  return Base::get(context, elementType, minNumElements);
}

LLVMScalableVectorType LLVMScalableVectorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, unsigned minNumElements) {
  return Base::getChecked(emitError, context, elementType, minNumElements);
}

::mlir::Type LLVMScalableVectorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_minNumElements;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '?'
  if (odsParser.parseQuestion()) return {};
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};

  // Parse variable 'minNumElements'
  _result_minNumElements = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_minNumElements)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMScalableVectorType parameter 'minNumElements' which is to be a `unsigned`");
    return {};
  }
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_elementType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_elementType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'elementType'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_minNumElements));
  return odsParser.getChecked<LLVMScalableVectorType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)),
      unsigned((*_result_minNumElements)));
}

void LLVMScalableVectorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "?";
  odsPrinter << ' ' << "x";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getMinNumElements());
  odsPrinter << ' ' << "x";
  odsPrinter << " ";
  odsPrinter << ' ';
  printPrettyLLVMType(odsPrinter,
    getElementType());
  odsPrinter << ">";
}

Type LLVMScalableVectorType::getElementType() const {
  return getImpl()->elementType;
}

unsigned LLVMScalableVectorType::getMinNumElements() const {
  return getImpl()->minNumElements;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMScalableVectorType)

#endif  // GET_TYPEDEF_CLASSES

