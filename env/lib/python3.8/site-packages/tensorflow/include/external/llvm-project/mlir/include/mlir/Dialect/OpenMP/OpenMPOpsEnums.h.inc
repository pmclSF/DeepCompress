/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
// CancellationConstructType Clause
enum class ClauseCancellationConstructType : uint32_t {
  Parallel = 0,
  Loop = 1,
  Sections = 2,
  Taskgroup = 3,
};

::std::optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(uint32_t);
::llvm::StringRef stringifyClauseCancellationConstructType(ClauseCancellationConstructType);
::std::optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseCancellationConstructType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ClauseCancellationConstructType enumValue) {
  return stringifyClauseCancellationConstructType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseCancellationConstructType> symbolizeEnum<ClauseCancellationConstructType>(::llvm::StringRef str) {
  return symbolizeClauseCancellationConstructType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseCancellationConstructType, ::mlir::omp::ClauseCancellationConstructType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseCancellationConstructType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for CancellationConstructType Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseCancellationConstructType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseCancellationConstructType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid CancellationConstructType Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseCancellationConstructType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseCancellationConstructType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseCancellationConstructType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseCancellationConstructType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseCancellationConstructType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseCancellationConstructType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseCancellationConstructType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseCancellationConstructType &lhs, const ::mlir::omp::ClauseCancellationConstructType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// depend clause
enum class ClauseDepend : uint32_t {
  dependsource = 0,
  dependsink = 1,
};

::std::optional<ClauseDepend> symbolizeClauseDepend(uint32_t);
::llvm::StringRef stringifyClauseDepend(ClauseDepend);
::std::optional<ClauseDepend> symbolizeClauseDepend(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseDepend() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseDepend enumValue) {
  return stringifyClauseDepend(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseDepend> symbolizeEnum<ClauseDepend>(::llvm::StringRef str) {
  return symbolizeClauseDepend(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseDepend, ::mlir::omp::ClauseDepend> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseDepend> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for depend clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseDepend> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseDepend>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid depend clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseDepend value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseDepend> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseDepend getEmptyKey() {
    return static_cast<::mlir::omp::ClauseDepend>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseDepend getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseDepend>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseDepend &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseDepend &lhs, const ::mlir::omp::ClauseDepend &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// task depend clause
enum class ClauseTaskDepend : uint32_t {
  taskdependin = 0,
  taskdependout = 1,
  taskdependinout = 2,
};

::std::optional<ClauseTaskDepend> symbolizeClauseTaskDepend(uint32_t);
::llvm::StringRef stringifyClauseTaskDepend(ClauseTaskDepend);
::std::optional<ClauseTaskDepend> symbolizeClauseTaskDepend(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseTaskDepend() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(ClauseTaskDepend enumValue) {
  return stringifyClauseTaskDepend(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseTaskDepend> symbolizeEnum<ClauseTaskDepend>(::llvm::StringRef str) {
  return symbolizeClauseTaskDepend(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseTaskDepend, ::mlir::omp::ClauseTaskDepend> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseTaskDepend> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for task depend clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseTaskDepend> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseTaskDepend>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid task depend clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseTaskDepend value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseTaskDepend> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseTaskDepend getEmptyKey() {
    return static_cast<::mlir::omp::ClauseTaskDepend>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseTaskDepend getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseTaskDepend>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseTaskDepend &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseTaskDepend &lhs, const ::mlir::omp::ClauseTaskDepend &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// GrainsizeType Clause
enum class ClauseGrainsizeType : uint32_t {
  Strict = 0,
};

::std::optional<ClauseGrainsizeType> symbolizeClauseGrainsizeType(uint32_t);
::llvm::StringRef stringifyClauseGrainsizeType(ClauseGrainsizeType);
::std::optional<ClauseGrainsizeType> symbolizeClauseGrainsizeType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseGrainsizeType() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(ClauseGrainsizeType enumValue) {
  return stringifyClauseGrainsizeType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseGrainsizeType> symbolizeEnum<ClauseGrainsizeType>(::llvm::StringRef str) {
  return symbolizeClauseGrainsizeType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseGrainsizeType, ::mlir::omp::ClauseGrainsizeType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseGrainsizeType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for GrainsizeType Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseGrainsizeType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseGrainsizeType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid GrainsizeType Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseGrainsizeType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseGrainsizeType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseGrainsizeType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseGrainsizeType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseGrainsizeType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseGrainsizeType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseGrainsizeType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseGrainsizeType &lhs, const ::mlir::omp::ClauseGrainsizeType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// MemoryOrderKind Clause
enum class ClauseMemoryOrderKind : uint32_t {
  Seq_cst = 0,
  Acq_rel = 1,
  Acquire = 2,
  Release = 3,
  Relaxed = 4,
};

::std::optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(uint32_t);
::llvm::StringRef stringifyClauseMemoryOrderKind(ClauseMemoryOrderKind);
::std::optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseMemoryOrderKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ClauseMemoryOrderKind enumValue) {
  return stringifyClauseMemoryOrderKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseMemoryOrderKind> symbolizeEnum<ClauseMemoryOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseMemoryOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseMemoryOrderKind, ::mlir::omp::ClauseMemoryOrderKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseMemoryOrderKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for MemoryOrderKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseMemoryOrderKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseMemoryOrderKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid MemoryOrderKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseMemoryOrderKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseMemoryOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseMemoryOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseMemoryOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseMemoryOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseMemoryOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseMemoryOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseMemoryOrderKind &lhs, const ::mlir::omp::ClauseMemoryOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// NumTasksType Clause
enum class ClauseNumTasksType : uint32_t {
  Strict = 0,
};

::std::optional<ClauseNumTasksType> symbolizeClauseNumTasksType(uint32_t);
::llvm::StringRef stringifyClauseNumTasksType(ClauseNumTasksType);
::std::optional<ClauseNumTasksType> symbolizeClauseNumTasksType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseNumTasksType() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(ClauseNumTasksType enumValue) {
  return stringifyClauseNumTasksType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseNumTasksType> symbolizeEnum<ClauseNumTasksType>(::llvm::StringRef str) {
  return symbolizeClauseNumTasksType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseNumTasksType, ::mlir::omp::ClauseNumTasksType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseNumTasksType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NumTasksType Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseNumTasksType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseNumTasksType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NumTasksType Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseNumTasksType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseNumTasksType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseNumTasksType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseNumTasksType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseNumTasksType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseNumTasksType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseNumTasksType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseNumTasksType &lhs, const ::mlir::omp::ClauseNumTasksType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OrderKind Clause
enum class ClauseOrderKind : uint32_t {
  Concurrent = 1,
};

::std::optional<ClauseOrderKind> symbolizeClauseOrderKind(uint32_t);
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind);
::std::optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseOrderKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseOrderKind enumValue) {
  return stringifyClauseOrderKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseOrderKind> symbolizeEnum<ClauseOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseOrderKind, ::mlir::omp::ClauseOrderKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseOrderKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for OrderKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseOrderKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseOrderKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid OrderKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseOrderKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseOrderKind &lhs, const ::mlir::omp::ClauseOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ProcBindKind Clause
enum class ClauseProcBindKind : uint32_t {
  Primary = 0,
  Master = 1,
  Close = 2,
  Spread = 3,
};

::std::optional<ClauseProcBindKind> symbolizeClauseProcBindKind(uint32_t);
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind);
::std::optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseProcBindKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ClauseProcBindKind enumValue) {
  return stringifyClauseProcBindKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseProcBindKind> symbolizeEnum<ClauseProcBindKind>(::llvm::StringRef str) {
  return symbolizeClauseProcBindKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseProcBindKind, ::mlir::omp::ClauseProcBindKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseProcBindKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ProcBindKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseProcBindKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseProcBindKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ProcBindKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseProcBindKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseProcBindKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseProcBindKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseProcBindKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseProcBindKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseProcBindKind &lhs, const ::mlir::omp::ClauseProcBindKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ScheduleKind Clause
enum class ClauseScheduleKind : uint32_t {
  Static = 0,
  Dynamic = 1,
  Guided = 2,
  Auto = 3,
  Runtime = 4,
};

::std::optional<ClauseScheduleKind> symbolizeClauseScheduleKind(uint32_t);
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind);
::std::optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseScheduleKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ClauseScheduleKind enumValue) {
  return stringifyClauseScheduleKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseScheduleKind> symbolizeEnum<ClauseScheduleKind>(::llvm::StringRef str) {
  return symbolizeClauseScheduleKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseScheduleKind, ::mlir::omp::ClauseScheduleKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseScheduleKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ScheduleKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseScheduleKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseScheduleKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ScheduleKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseScheduleKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseScheduleKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseScheduleKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseScheduleKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseScheduleKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseScheduleKind &lhs, const ::mlir::omp::ClauseScheduleKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OpenMP Schedule Modifier
enum class ScheduleModifier : uint32_t {
  none = 0,
  monotonic = 1,
  nonmonotonic = 2,
  simd = 3,
};

::std::optional<ScheduleModifier> symbolizeScheduleModifier(uint32_t);
::llvm::StringRef stringifyScheduleModifier(ScheduleModifier);
::std::optional<ScheduleModifier> symbolizeScheduleModifier(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForScheduleModifier() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ScheduleModifier enumValue) {
  return stringifyScheduleModifier(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ScheduleModifier> symbolizeEnum<ScheduleModifier>(::llvm::StringRef str) {
  return symbolizeScheduleModifier(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ScheduleModifier, ::mlir::omp::ScheduleModifier> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ScheduleModifier> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for OpenMP Schedule Modifier");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ScheduleModifier> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ScheduleModifier>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid OpenMP Schedule Modifier specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ScheduleModifier value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ScheduleModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ScheduleModifier getEmptyKey() {
    return static_cast<::mlir::omp::ScheduleModifier>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ScheduleModifier getTombstoneKey() {
    return static_cast<::mlir::omp::ScheduleModifier>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ScheduleModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ScheduleModifier &lhs, const ::mlir::omp::ScheduleModifier &rhs) {
    return lhs == rhs;
  }
};
}

