/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace NVVM {
// MMA binary operations
enum class MMAB1Op : uint32_t {
  none = 0,
  xor_popc = 1,
  and_popc = 2,
};

::std::optional<MMAB1Op> symbolizeMMAB1Op(uint32_t);
::llvm::StringRef stringifyMMAB1Op(MMAB1Op);
::std::optional<MMAB1Op> symbolizeMMAB1Op(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAB1Op() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(MMAB1Op enumValue) {
  return stringifyMMAB1Op(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAB1Op> symbolizeEnum<MMAB1Op>(::llvm::StringRef str) {
  return symbolizeMMAB1Op(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMAB1Op, ::mlir::NVVM::MMAB1Op> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMAB1Op> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for MMA binary operations");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMAB1Op> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMAB1Op>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid MMA binary operations specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMAB1Op value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAB1Op> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAB1Op getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAB1Op>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAB1Op getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAB1Op>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAB1Op &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAB1Op &lhs, const ::mlir::NVVM::MMAB1Op &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA frag type
enum class MMAFrag : uint32_t {
  a = 0,
  b = 1,
  c = 2,
};

::std::optional<MMAFrag> symbolizeMMAFrag(uint32_t);
::llvm::StringRef stringifyMMAFrag(MMAFrag);
::std::optional<MMAFrag> symbolizeMMAFrag(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAFrag() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(MMAFrag enumValue) {
  return stringifyMMAFrag(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAFrag> symbolizeEnum<MMAFrag>(::llvm::StringRef str) {
  return symbolizeMMAFrag(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMAFrag, ::mlir::NVVM::MMAFrag> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMAFrag> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM MMA frag type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMAFrag> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMAFrag>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM MMA frag type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMAFrag value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAFrag> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAFrag getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAFrag>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAFrag getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAFrag>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAFrag &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAFrag &lhs, const ::mlir::NVVM::MMAFrag &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// MMA overflow options
enum class MMAIntOverflow : uint32_t {
  satfinite = 1,
  wrapped = 0,
};

::std::optional<MMAIntOverflow> symbolizeMMAIntOverflow(uint32_t);
::llvm::StringRef stringifyMMAIntOverflow(MMAIntOverflow);
::std::optional<MMAIntOverflow> symbolizeMMAIntOverflow(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAIntOverflow() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MMAIntOverflow enumValue) {
  return stringifyMMAIntOverflow(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAIntOverflow> symbolizeEnum<MMAIntOverflow>(::llvm::StringRef str) {
  return symbolizeMMAIntOverflow(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMAIntOverflow, ::mlir::NVVM::MMAIntOverflow> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMAIntOverflow> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for MMA overflow options");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMAIntOverflow> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMAIntOverflow>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid MMA overflow options specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMAIntOverflow value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAIntOverflow> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAIntOverflow getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAIntOverflow>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAIntOverflow getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAIntOverflow>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAIntOverflow &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAIntOverflow &lhs, const ::mlir::NVVM::MMAIntOverflow &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA layout
enum class MMALayout : uint32_t {
  row = 0,
  col = 1,
};

::std::optional<MMALayout> symbolizeMMALayout(uint32_t);
::llvm::StringRef stringifyMMALayout(MMALayout);
::std::optional<MMALayout> symbolizeMMALayout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMALayout() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MMALayout enumValue) {
  return stringifyMMALayout(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMALayout> symbolizeEnum<MMALayout>(::llvm::StringRef str) {
  return symbolizeMMALayout(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMALayout, ::mlir::NVVM::MMALayout> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMALayout> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM MMA layout");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMALayout> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMALayout>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM MMA layout specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMALayout value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMALayout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMALayout getEmptyKey() {
    return static_cast<::mlir::NVVM::MMALayout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMALayout getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMALayout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMALayout &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMALayout &lhs, const ::mlir::NVVM::MMALayout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA types
enum class MMATypes : uint32_t {
  f16 = 0,
  f32 = 1,
  tf32 = 2,
  bf16 = 9,
  s8 = 4,
  u8 = 3,
  s32 = 5,
  s4 = 8,
  u4 = 7,
  b1 = 6,
  f64 = 10,
};

::std::optional<MMATypes> symbolizeMMATypes(uint32_t);
::llvm::StringRef stringifyMMATypes(MMATypes);
::std::optional<MMATypes> symbolizeMMATypes(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMATypes() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(MMATypes enumValue) {
  return stringifyMMATypes(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMATypes> symbolizeEnum<MMATypes>(::llvm::StringRef str) {
  return symbolizeMMATypes(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMATypes, ::mlir::NVVM::MMATypes> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMATypes> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM MMA types");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMATypes> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMATypes>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM MMA types specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMATypes value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMATypes> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMATypes getEmptyKey() {
    return static_cast<::mlir::NVVM::MMATypes>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMATypes getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMATypes>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMATypes &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMATypes &lhs, const ::mlir::NVVM::MMATypes &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM redux kind
enum class ReduxKind : uint32_t {
  ADD = 1,
  AND = 2,
  MAX = 3,
  MIN = 4,
  OR = 5,
  UMAX = 6,
  UMIN = 7,
  XOR = 8,
};

::std::optional<ReduxKind> symbolizeReduxKind(uint32_t);
::llvm::StringRef stringifyReduxKind(ReduxKind);
::std::optional<ReduxKind> symbolizeReduxKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReduxKind() {
  return 8;
}


inline ::llvm::StringRef stringifyEnum(ReduxKind enumValue) {
  return stringifyReduxKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReduxKind> symbolizeEnum<ReduxKind>(::llvm::StringRef str) {
  return symbolizeReduxKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::ReduxKind, ::mlir::NVVM::ReduxKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::ReduxKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM redux kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::ReduxKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::ReduxKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM redux kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::ReduxKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::ReduxKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::ReduxKind getEmptyKey() {
    return static_cast<::mlir::NVVM::ReduxKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::ReduxKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::ReduxKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::ReduxKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::ReduxKind &lhs, const ::mlir::NVVM::ReduxKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM shuffle kind
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};

::std::optional<ShflKind> symbolizeShflKind(uint32_t);
::llvm::StringRef stringifyShflKind(ShflKind);
::std::optional<ShflKind> symbolizeShflKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForShflKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ShflKind enumValue) {
  return stringifyShflKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ShflKind> symbolizeEnum<ShflKind>(::llvm::StringRef str) {
  return symbolizeShflKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::ShflKind, ::mlir::NVVM::ShflKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::ShflKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM shuffle kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::ShflKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::ShflKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM shuffle kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::ShflKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::ShflKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::ShflKind getEmptyKey() {
    return static_cast<::mlir::NVVM::ShflKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::ShflKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::ShflKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::ShflKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::ShflKind &lhs, const ::mlir::NVVM::ShflKind &rhs) {
    return lhs == rhs;
  }
};
}

