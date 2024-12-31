/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
// threads for loop mapping
enum class Blocks : uint64_t {
  DimX = 0,
  DimY = 1,
  DimZ = 2,
};

::std::optional<Blocks> symbolizeBlocks(uint64_t);
::llvm::StringRef stringifyBlocks(Blocks);
::std::optional<Blocks> symbolizeBlocks(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBlocks() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Blocks enumValue) {
  return stringifyBlocks(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Blocks> symbolizeEnum<Blocks>(::llvm::StringRef str) {
  return symbolizeBlocks(str);
}

class BlocksAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Blocks;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static BlocksAttr get(::mlir::MLIRContext *context, Blocks val);
  Blocks getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Blocks, ::mlir::gpu::Blocks> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Blocks> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for threads for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Blocks> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Blocks>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid threads for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Blocks value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Blocks> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::Blocks getEmptyKey() {
    return static_cast<::mlir::gpu::Blocks>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Blocks getTombstoneKey() {
    return static_cast<::mlir::gpu::Blocks>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Blocks &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Blocks &lhs, const ::mlir::gpu::Blocks &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// GPU address space
enum class AddressSpace : uint32_t {
  Global = 1,
  Workgroup = 2,
  Private = 3,
};

::std::optional<AddressSpace> symbolizeAddressSpace(uint32_t);
::llvm::StringRef stringifyAddressSpace(AddressSpace);
::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAddressSpace() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(AddressSpace enumValue) {
  return stringifyAddressSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AddressSpace> symbolizeEnum<AddressSpace>(::llvm::StringRef str) {
  return symbolizeAddressSpace(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::AddressSpace, ::mlir::gpu::AddressSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::AddressSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for GPU address space");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::AddressSpace> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::AddressSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid GPU address space specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::AddressSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::AddressSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::AddressSpace getEmptyKey() {
    return static_cast<::mlir::gpu::AddressSpace>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::AddressSpace getTombstoneKey() {
    return static_cast<::mlir::gpu::AddressSpace>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::AddressSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::AddressSpace &lhs, const ::mlir::gpu::AddressSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// built-in reduction operations supported by gpu.allreduce.
enum class AllReduceOperation : uint32_t {
  ADD = 0,
  AND = 1,
  MAX = 2,
  MIN = 3,
  MUL = 4,
  OR = 5,
  XOR = 6,
};

::std::optional<AllReduceOperation> symbolizeAllReduceOperation(uint32_t);
::llvm::StringRef stringifyAllReduceOperation(AllReduceOperation);
::std::optional<AllReduceOperation> symbolizeAllReduceOperation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAllReduceOperation() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(AllReduceOperation enumValue) {
  return stringifyAllReduceOperation(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AllReduceOperation> symbolizeEnum<AllReduceOperation>(::llvm::StringRef str) {
  return symbolizeAllReduceOperation(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::AllReduceOperation, ::mlir::gpu::AllReduceOperation> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::AllReduceOperation> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for built-in reduction operations supported by gpu.allreduce.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::AllReduceOperation> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::AllReduceOperation>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid built-in reduction operations supported by gpu.allreduce. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::AllReduceOperation value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::AllReduceOperation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::AllReduceOperation getEmptyKey() {
    return static_cast<::mlir::gpu::AllReduceOperation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::AllReduceOperation getTombstoneKey() {
    return static_cast<::mlir::gpu::AllReduceOperation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::AllReduceOperation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::AllReduceOperation &lhs, const ::mlir::gpu::AllReduceOperation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// a dimension, either 'x', 'y', or 'z'
enum class Dimension : uint32_t {
  x = 0,
  y = 1,
  z = 2,
};

::std::optional<Dimension> symbolizeDimension(uint32_t);
::llvm::StringRef stringifyDimension(Dimension);
::std::optional<Dimension> symbolizeDimension(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDimension() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Dimension enumValue) {
  return stringifyDimension(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Dimension> symbolizeEnum<Dimension>(::llvm::StringRef str) {
  return symbolizeDimension(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Dimension, ::mlir::gpu::Dimension> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Dimension> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for a dimension, either 'x', 'y', or 'z'");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Dimension> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Dimension>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid a dimension, either 'x', 'y', or 'z' specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Dimension value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Dimension> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::Dimension getEmptyKey() {
    return static_cast<::mlir::gpu::Dimension>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Dimension getTombstoneKey() {
    return static_cast<::mlir::gpu::Dimension>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Dimension &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Dimension &lhs, const ::mlir::gpu::Dimension &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// Indexing modes supported by gpu.shuffle.
enum class ShuffleMode : uint32_t {
  XOR = 0,
  UP = 2,
  DOWN = 1,
  IDX = 3,
};

::std::optional<ShuffleMode> symbolizeShuffleMode(uint32_t);
::llvm::StringRef stringifyShuffleMode(ShuffleMode);
::std::optional<ShuffleMode> symbolizeShuffleMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForShuffleMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ShuffleMode enumValue) {
  return stringifyShuffleMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ShuffleMode> symbolizeEnum<ShuffleMode>(::llvm::StringRef str) {
  return symbolizeShuffleMode(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::ShuffleMode, ::mlir::gpu::ShuffleMode> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::ShuffleMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Indexing modes supported by gpu.shuffle.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::ShuffleMode> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::ShuffleMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Indexing modes supported by gpu.shuffle. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::ShuffleMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::ShuffleMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::ShuffleMode getEmptyKey() {
    return static_cast<::mlir::gpu::ShuffleMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::ShuffleMode getTombstoneKey() {
    return static_cast<::mlir::gpu::ShuffleMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::ShuffleMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::ShuffleMode &lhs, const ::mlir::gpu::ShuffleMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// linear ids for loop mapping
enum class LinearId : uint64_t {
  DimX = 0,
  DimY = 1,
  DimZ = 2,
};

::std::optional<LinearId> symbolizeLinearId(uint64_t);
::llvm::StringRef stringifyLinearId(LinearId);
::std::optional<LinearId> symbolizeLinearId(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLinearId() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(LinearId enumValue) {
  return stringifyLinearId(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<LinearId> symbolizeEnum<LinearId>(::llvm::StringRef str) {
  return symbolizeLinearId(str);
}

class LinearIdAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = LinearId;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LinearIdAttr get(::mlir::MLIRContext *context, LinearId val);
  LinearId getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::LinearId, ::mlir::gpu::LinearId> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::LinearId> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for linear ids for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::LinearId> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::LinearId>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid linear ids for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::LinearId value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::LinearId> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::LinearId getEmptyKey() {
    return static_cast<::mlir::gpu::LinearId>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::LinearId getTombstoneKey() {
    return static_cast<::mlir::gpu::LinearId>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::LinearId &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::LinearId &lhs, const ::mlir::gpu::LinearId &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// elementwise operation to apply to mma matrix
enum class MMAElementwiseOp : uint32_t {
  ADDF = 0,
  MULF = 1,
  SUBF = 2,
  MAXF = 3,
  MINF = 4,
  DIVF = 5,
  ADDI = 6,
  MULI = 7,
  SUBI = 8,
  DIVS = 9,
  DIVU = 10,
  NEGATEF = 11,
  NEGATES = 12,
};

::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(uint32_t);
::llvm::StringRef stringifyMMAElementwiseOp(MMAElementwiseOp);
::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAElementwiseOp() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(MMAElementwiseOp enumValue) {
  return stringifyMMAElementwiseOp(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAElementwiseOp> symbolizeEnum<MMAElementwiseOp>(::llvm::StringRef str) {
  return symbolizeMMAElementwiseOp(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::MMAElementwiseOp, ::mlir::gpu::MMAElementwiseOp> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::MMAElementwiseOp> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for elementwise operation to apply to mma matrix");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::MMAElementwiseOp> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::MMAElementwiseOp>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid elementwise operation to apply to mma matrix specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::MMAElementwiseOp value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::MMAElementwiseOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::MMAElementwiseOp getEmptyKey() {
    return static_cast<::mlir::gpu::MMAElementwiseOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::MMAElementwiseOp getTombstoneKey() {
    return static_cast<::mlir::gpu::MMAElementwiseOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::MMAElementwiseOp &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::MMAElementwiseOp &lhs, const ::mlir::gpu::MMAElementwiseOp &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// processor for loop mapping
enum class Processor : uint64_t {
  BlockX = 0,
  BlockY = 1,
  BlockZ = 2,
  ThreadX = 3,
  ThreadY = 4,
  ThreadZ = 5,
  Sequential = 6,
};

::std::optional<Processor> symbolizeProcessor(uint64_t);
::llvm::StringRef stringifyProcessor(Processor);
::std::optional<Processor> symbolizeProcessor(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForProcessor() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(Processor enumValue) {
  return stringifyProcessor(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Processor> symbolizeEnum<Processor>(::llvm::StringRef str) {
  return symbolizeProcessor(str);
}

class ProcessorAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Processor;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ProcessorAttr get(::mlir::MLIRContext *context, Processor val);
  Processor getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Processor, ::mlir::gpu::Processor> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Processor> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for processor for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Processor> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Processor>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid processor for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Processor value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Processor> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::Processor getEmptyKey() {
    return static_cast<::mlir::gpu::Processor>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Processor getTombstoneKey() {
    return static_cast<::mlir::gpu::Processor>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Processor &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Processor &lhs, const ::mlir::gpu::Processor &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// threads for loop mapping
enum class Threads : uint64_t {
  DimX = 0,
  DimY = 1,
  DimZ = 2,
};

::std::optional<Threads> symbolizeThreads(uint64_t);
::llvm::StringRef stringifyThreads(Threads);
::std::optional<Threads> symbolizeThreads(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForThreads() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Threads enumValue) {
  return stringifyThreads(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Threads> symbolizeEnum<Threads>(::llvm::StringRef str) {
  return symbolizeThreads(str);
}

class ThreadsAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Threads;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ThreadsAttr get(::mlir::MLIRContext *context, Threads val);
  Threads getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Threads, ::mlir::gpu::Threads> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Threads> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for threads for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Threads> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Threads>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid threads for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Threads value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Threads> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::Threads getEmptyKey() {
    return static_cast<::mlir::gpu::Threads>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Threads getTombstoneKey() {
    return static_cast<::mlir::gpu::Threads>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Threads &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Threads &lhs, const ::mlir::gpu::Threads &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// threads for loop mapping
enum class Warps : uint64_t {
  DimX = 0,
  DimY = 1,
  DimZ = 2,
};

::std::optional<Warps> symbolizeWarps(uint64_t);
::llvm::StringRef stringifyWarps(Warps);
::std::optional<Warps> symbolizeWarps(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForWarps() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Warps enumValue) {
  return stringifyWarps(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Warps> symbolizeEnum<Warps>(::llvm::StringRef str) {
  return symbolizeWarps(str);
}

class WarpsAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Warps;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static WarpsAttr get(::mlir::MLIRContext *context, Warps val);
  Warps getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Warps, ::mlir::gpu::Warps> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Warps> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for threads for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Warps> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Warps>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid threads for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Warps value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Warps> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::Warps getEmptyKey() {
    return static_cast<::mlir::gpu::Warps>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Warps getTombstoneKey() {
    return static_cast<::mlir::gpu::Warps>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Warps &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Warps &lhs, const ::mlir::gpu::Warps &rhs) {
    return lhs == rhs;
  }
};
}

