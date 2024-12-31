/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::gpu::GPUBlockMappingAttr,
::mlir::gpu::GPULinearIdMappingAttr,
::mlir::gpu::GPUMemorySpaceMappingAttr,
::mlir::gpu::GPUThreadMappingAttr,
::mlir::gpu::GPUWarpMappingAttr,
::mlir::gpu::AddressSpaceAttr,
::mlir::gpu::AllReduceOperationAttr,
::mlir::gpu::DimensionAttr,
::mlir::gpu::ShuffleModeAttr,
::mlir::gpu::MMAElementwiseOpAttr,
::mlir::gpu::ParallelLoopDimMappingAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::gpu::GPUBlockMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUBlockMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPULinearIdMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPULinearIdMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUMemorySpaceMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUMemorySpaceMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUThreadMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUThreadMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUWarpMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUWarpMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::AddressSpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::AddressSpaceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::AllReduceOperationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::AllReduceOperationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::DimensionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::DimensionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ShuffleModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ShuffleModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::MMAElementwiseOpAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::MMAElementwiseOpAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ParallelLoopDimMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ParallelLoopDimMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::gpu::GPUBlockMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUBlockMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPULinearIdMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPULinearIdMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUMemorySpaceMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUMemorySpaceMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUThreadMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUThreadMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUWarpMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUWarpMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::AddressSpaceAttr>([&](auto t) {
      printer << ::mlir::gpu::AddressSpaceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::AllReduceOperationAttr>([&](auto t) {
      printer << ::mlir::gpu::AllReduceOperationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::DimensionAttr>([&](auto t) {
      printer << ::mlir::gpu::DimensionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ShuffleModeAttr>([&](auto t) {
      printer << ::mlir::gpu::ShuffleModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::MMAElementwiseOpAttr>([&](auto t) {
      printer << ::mlir::gpu::MMAElementwiseOpAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ParallelLoopDimMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::ParallelLoopDimMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace gpu {
namespace detail {
struct GPUBlockMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Blocks>;
  GPUBlockMappingAttrStorage(::mlir::gpu::Blocks block) : block(block) {}

  KeyTy getAsKey() const {
    return KeyTy(block);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (block == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUBlockMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto block = std::get<0>(tblgenKey);
    return new (allocator.allocate<GPUBlockMappingAttrStorage>()) GPUBlockMappingAttrStorage(block);
  }

  ::mlir::gpu::Blocks block;
};
} // namespace detail
GPUBlockMappingAttr GPUBlockMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Blocks block) {
  return Base::get(context, block);
}

::mlir::Attribute GPUBlockMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Blocks> _result_block;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'block'
  _result_block = [&]() -> ::mlir::FailureOr<::mlir::gpu::Blocks> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeBlocks(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Blocks" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_block)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUBlockMappingAttr parameter 'block' which is to be a `::mlir::gpu::Blocks`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_block));
  return GPUBlockMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::Blocks((*_result_block)));
}

void GPUBlockMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyBlocks(getBlock());
  }
  odsPrinter << ">";
}

::mlir::gpu::Blocks GPUBlockMappingAttr::getBlock() const {
  return getImpl()->block;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUBlockMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPULinearIdMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::LinearId>;
  GPULinearIdMappingAttrStorage(::mlir::gpu::LinearId linear_id) : linear_id(linear_id) {}

  KeyTy getAsKey() const {
    return KeyTy(linear_id);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (linear_id == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPULinearIdMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto linear_id = std::get<0>(tblgenKey);
    return new (allocator.allocate<GPULinearIdMappingAttrStorage>()) GPULinearIdMappingAttrStorage(linear_id);
  }

  ::mlir::gpu::LinearId linear_id;
};
} // namespace detail
GPULinearIdMappingAttr GPULinearIdMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::LinearId linear_id) {
  return Base::get(context, linear_id);
}

::mlir::Attribute GPULinearIdMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::LinearId> _result_linear_id;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'linear_id'
  _result_linear_id = [&]() -> ::mlir::FailureOr<::mlir::gpu::LinearId> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeLinearId(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::LinearId" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_linear_id)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPULinearIdMapping parameter 'linear_id' which is to be a `::mlir::gpu::LinearId`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_linear_id));
  return GPULinearIdMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::LinearId((*_result_linear_id)));
}

void GPULinearIdMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyLinearId(getLinearId());
  }
  odsPrinter << ">";
}

::mlir::gpu::LinearId GPULinearIdMappingAttr::getLinearId() const {
  return getImpl()->linear_id;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPULinearIdMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUMemorySpaceMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AddressSpace>;
  GPUMemorySpaceMappingAttrStorage(::mlir::gpu::AddressSpace address_space) : address_space(address_space) {}

  KeyTy getAsKey() const {
    return KeyTy(address_space);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (address_space == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUMemorySpaceMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto address_space = std::get<0>(tblgenKey);
    return new (allocator.allocate<GPUMemorySpaceMappingAttrStorage>()) GPUMemorySpaceMappingAttrStorage(address_space);
  }

  ::mlir::gpu::AddressSpace address_space;
};
} // namespace detail
GPUMemorySpaceMappingAttr GPUMemorySpaceMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace address_space) {
  return Base::get(context, address_space);
}

::mlir::Attribute GPUMemorySpaceMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::AddressSpace> _result_address_space;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'address_space'
  _result_address_space = [&]() -> ::mlir::FailureOr<::mlir::gpu::AddressSpace> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAddressSpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AddressSpace" << " to be one of: " << "global" << ", " << "workgroup" << ", " << "private")};
    }();
  if (::mlir::failed(_result_address_space)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUMemorySpaceMappingAttr parameter 'address_space' which is to be a `::mlir::gpu::AddressSpace`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_address_space));
  return GPUMemorySpaceMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::AddressSpace((*_result_address_space)));
}

void GPUMemorySpaceMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyAddressSpace(getAddressSpace());
  }
  odsPrinter << ">";
}

::mlir::gpu::AddressSpace GPUMemorySpaceMappingAttr::getAddressSpace() const {
  return getImpl()->address_space;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUMemorySpaceMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUThreadMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Threads>;
  GPUThreadMappingAttrStorage(::mlir::gpu::Threads thread) : thread(thread) {}

  KeyTy getAsKey() const {
    return KeyTy(thread);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (thread == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUThreadMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto thread = std::get<0>(tblgenKey);
    return new (allocator.allocate<GPUThreadMappingAttrStorage>()) GPUThreadMappingAttrStorage(thread);
  }

  ::mlir::gpu::Threads thread;
};
} // namespace detail
GPUThreadMappingAttr GPUThreadMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Threads thread) {
  return Base::get(context, thread);
}

::mlir::Attribute GPUThreadMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Threads> _result_thread;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'thread'
  _result_thread = [&]() -> ::mlir::FailureOr<::mlir::gpu::Threads> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeThreads(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Threads" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_thread)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUThreadMappingAttr parameter 'thread' which is to be a `::mlir::gpu::Threads`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_thread));
  return GPUThreadMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::Threads((*_result_thread)));
}

void GPUThreadMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyThreads(getThread());
  }
  odsPrinter << ">";
}

::mlir::gpu::Threads GPUThreadMappingAttr::getThread() const {
  return getImpl()->thread;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUThreadMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUWarpMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Warps>;
  GPUWarpMappingAttrStorage(::mlir::gpu::Warps warp) : warp(warp) {}

  KeyTy getAsKey() const {
    return KeyTy(warp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (warp == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUWarpMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto warp = std::get<0>(tblgenKey);
    return new (allocator.allocate<GPUWarpMappingAttrStorage>()) GPUWarpMappingAttrStorage(warp);
  }

  ::mlir::gpu::Warps warp;
};
} // namespace detail
GPUWarpMappingAttr GPUWarpMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Warps warp) {
  return Base::get(context, warp);
}

::mlir::Attribute GPUWarpMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Warps> _result_warp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'warp'
  _result_warp = [&]() -> ::mlir::FailureOr<::mlir::gpu::Warps> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeWarps(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Warps" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_warp)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUWarpMappingAttr parameter 'warp' which is to be a `::mlir::gpu::Warps`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_warp));
  return GPUWarpMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::Warps((*_result_warp)));
}

void GPUWarpMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyWarps(getWarp());
  }
  odsPrinter << ">";
}

::mlir::gpu::Warps GPUWarpMappingAttr::getWarp() const {
  return getImpl()->warp;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUWarpMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct AddressSpaceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AddressSpace>;
  AddressSpaceAttrStorage(::mlir::gpu::AddressSpace value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AddressSpaceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<AddressSpaceAttrStorage>()) AddressSpaceAttrStorage(value);
  }

  ::mlir::gpu::AddressSpace value;
};
} // namespace detail
AddressSpaceAttr AddressSpaceAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace value) {
  return Base::get(context, value);
}

::mlir::Attribute AddressSpaceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::AddressSpace> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::AddressSpace> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAddressSpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AddressSpace" << " to be one of: " << "global" << ", " << "workgroup" << ", " << "private")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_AddressSpaceAttr parameter 'value' which is to be a `::mlir::gpu::AddressSpace`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return AddressSpaceAttr::get(odsParser.getContext(),
      ::mlir::gpu::AddressSpace((*_result_value)));
}

void AddressSpaceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyAddressSpace(getValue());
  odsPrinter << ">";
}

::mlir::gpu::AddressSpace AddressSpaceAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AddressSpaceAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct AllReduceOperationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AllReduceOperation>;
  AllReduceOperationAttrStorage(::mlir::gpu::AllReduceOperation value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AllReduceOperationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<AllReduceOperationAttrStorage>()) AllReduceOperationAttrStorage(value);
  }

  ::mlir::gpu::AllReduceOperation value;
};
} // namespace detail
AllReduceOperationAttr AllReduceOperationAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AllReduceOperation value) {
  return Base::get(context, value);
}

::mlir::Attribute AllReduceOperationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::AllReduceOperation> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::AllReduceOperation> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAllReduceOperation(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AllReduceOperation" << " to be one of: " << "add" << ", " << "and" << ", " << "max" << ", " << "min" << ", " << "mul" << ", " << "or" << ", " << "xor")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_AllReduceOperationAttr parameter 'value' which is to be a `::mlir::gpu::AllReduceOperation`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return AllReduceOperationAttr::get(odsParser.getContext(),
      ::mlir::gpu::AllReduceOperation((*_result_value)));
}

void AllReduceOperationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyAllReduceOperation(getValue());
}

::mlir::gpu::AllReduceOperation AllReduceOperationAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct DimensionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Dimension>;
  DimensionAttrStorage(::mlir::gpu::Dimension value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DimensionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<DimensionAttrStorage>()) DimensionAttrStorage(value);
  }

  ::mlir::gpu::Dimension value;
};
} // namespace detail
DimensionAttr DimensionAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Dimension value) {
  return Base::get(context, value);
}

::mlir::Attribute DimensionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Dimension> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::Dimension> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeDimension(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Dimension" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_DimensionAttr parameter 'value' which is to be a `::mlir::gpu::Dimension`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return DimensionAttr::get(odsParser.getContext(),
      ::mlir::gpu::Dimension((*_result_value)));
}

void DimensionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyDimension(getValue());
}

::mlir::gpu::Dimension DimensionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ShuffleModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::ShuffleMode>;
  ShuffleModeAttrStorage(::mlir::gpu::ShuffleMode value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ShuffleModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ShuffleModeAttrStorage>()) ShuffleModeAttrStorage(value);
  }

  ::mlir::gpu::ShuffleMode value;
};
} // namespace detail
ShuffleModeAttr ShuffleModeAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::ShuffleMode value) {
  return Base::get(context, value);
}

::mlir::Attribute ShuffleModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::ShuffleMode> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::ShuffleMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeShuffleMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::ShuffleMode" << " to be one of: " << "xor" << ", " << "up" << ", " << "down" << ", " << "idx")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ShuffleModeAttr parameter 'value' which is to be a `::mlir::gpu::ShuffleMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ShuffleModeAttr::get(odsParser.getContext(),
      ::mlir::gpu::ShuffleMode((*_result_value)));
}

void ShuffleModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyShuffleMode(getValue());
}

::mlir::gpu::ShuffleMode ShuffleModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct MMAElementwiseOpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MMAElementwiseOp>;
  MMAElementwiseOpAttrStorage(::mlir::gpu::MMAElementwiseOp value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMAElementwiseOpAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMAElementwiseOpAttrStorage>()) MMAElementwiseOpAttrStorage(value);
  }

  ::mlir::gpu::MMAElementwiseOp value;
};
} // namespace detail
MMAElementwiseOpAttr MMAElementwiseOpAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MMAElementwiseOp value) {
  return Base::get(context, value);
}

::mlir::Attribute MMAElementwiseOpAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::MMAElementwiseOp> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::MMAElementwiseOp> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMMAElementwiseOp(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MMAElementwiseOp" << " to be one of: " << "addf" << ", " << "mulf" << ", " << "subf" << ", " << "maxf" << ", " << "minf" << ", " << "divf" << ", " << "addi" << ", " << "muli" << ", " << "subi" << ", " << "divs" << ", " << "divu" << ", " << "negatef" << ", " << "negates")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAElementWiseAttr parameter 'value' which is to be a `::mlir::gpu::MMAElementwiseOp`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MMAElementwiseOpAttr::get(odsParser.getContext(),
      ::mlir::gpu::MMAElementwiseOp((*_result_value)));
}

void MMAElementwiseOpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyMMAElementwiseOp(getValue());
}

::mlir::gpu::MMAElementwiseOp MMAElementwiseOpAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ParallelLoopDimMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Processor, AffineMap, AffineMap>;
  ParallelLoopDimMappingAttrStorage(::mlir::gpu::Processor processor, AffineMap map, AffineMap bound) : processor(processor), map(map), bound(bound) {}

  KeyTy getAsKey() const {
    return KeyTy(processor, map, bound);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (processor == std::get<0>(tblgenKey)) && (map == std::get<1>(tblgenKey)) && (bound == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static ParallelLoopDimMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto processor = std::get<0>(tblgenKey);
    auto map = std::get<1>(tblgenKey);
    auto bound = std::get<2>(tblgenKey);
    return new (allocator.allocate<ParallelLoopDimMappingAttrStorage>()) ParallelLoopDimMappingAttrStorage(processor, map, bound);
  }

  ::mlir::gpu::Processor processor;
  AffineMap map;
  AffineMap bound;
};
} // namespace detail
ParallelLoopDimMappingAttr ParallelLoopDimMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Processor processor, AffineMap map, AffineMap bound) {
  return Base::get(context, processor, map, bound);
}

::mlir::Attribute ParallelLoopDimMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Processor> _result_processor;
  ::mlir::FailureOr<AffineMap> _result_map;
  ::mlir::FailureOr<AffineMap> _result_bound;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_processor = false;
  bool _seen_map = false;
  bool _seen_bound = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_processor && _paramKey == "processor") {
        _seen_processor = true;

        // Parse variable 'processor'
        _result_processor = [&]() -> ::mlir::FailureOr<::mlir::gpu::Processor> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::gpu::symbolizeProcessor(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Processor" << " to be one of: " << "block_x" << ", " << "block_y" << ", " << "block_z" << ", " << "thread_x" << ", " << "thread_y" << ", " << "thread_z" << ", " << "sequential")};
          }();
        if (::mlir::failed(_result_processor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'processor' which is to be a `::mlir::gpu::Processor`");
          return {};
        }
      } else if (!_seen_map && _paramKey == "map") {
        _seen_map = true;

        // Parse variable 'map'
        _result_map = ::mlir::FieldParser<AffineMap>::parse(odsParser);
        if (::mlir::failed(_result_map)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'map' which is to be a `AffineMap`");
          return {};
        }
      } else if (!_seen_bound && _paramKey == "bound") {
        _seen_bound = true;

        // Parse variable 'bound'
        _result_bound = ::mlir::FieldParser<AffineMap>::parse(odsParser);
        if (::mlir::failed(_result_bound)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'bound' which is to be a `AffineMap`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 3; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 3 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_processor));
  assert(::mlir::succeeded(_result_map));
  assert(::mlir::succeeded(_result_bound));
  return ParallelLoopDimMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::Processor((*_result_processor)),
      AffineMap((*_result_map)),
      AffineMap((*_result_bound)));
}

void ParallelLoopDimMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "processor = ";
    odsPrinter << stringifyProcessor(getProcessor());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "map = ";
    odsPrinter.printStrippedAttrOrType(getMap());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "bound = ";
    odsPrinter.printStrippedAttrOrType(getBound());
  }
  odsPrinter << ">";
}

::mlir::gpu::Processor ParallelLoopDimMappingAttr::getProcessor() const {
  return getImpl()->processor;
}

AffineMap ParallelLoopDimMappingAttr::getMap() const {
  return getImpl()->map;
}

AffineMap ParallelLoopDimMappingAttr::getBound() const {
  return getImpl()->bound;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ParallelLoopDimMappingAttr)
namespace mlir {
namespace gpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute GPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void GPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace gpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

