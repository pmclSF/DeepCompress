/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
// DefaultValue Clause
enum class ClauseDefaultValue : uint32_t {
  Present = 0,
  None = 1,
};

::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(uint32_t);
::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue);
::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseDefaultValue() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseDefaultValue enumValue) {
  return stringifyClauseDefaultValue(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseDefaultValue> symbolizeEnum<ClauseDefaultValue>(::llvm::StringRef str) {
  return symbolizeClauseDefaultValue(str);
}
} // namespace acc
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::ClauseDefaultValue, ::mlir::acc::ClauseDefaultValue> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::ClauseDefaultValue> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for DefaultValue Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ClauseDefaultValue> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ClauseDefaultValue>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid DefaultValue Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::ClauseDefaultValue value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ClauseDefaultValue> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ClauseDefaultValue getEmptyKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ClauseDefaultValue getTombstoneKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ClauseDefaultValue &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ClauseDefaultValue &lhs, const ::mlir::acc::ClauseDefaultValue &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace acc {
// data clauses supported by OpenACC
enum class DataClause : uint64_t {
  acc_copyin = 1,
  acc_copyin_readonly = 2,
  acc_copy = 3,
  acc_copyout = 4,
  acc_copyout_zero = 5,
  acc_present = 6,
  acc_create = 7,
  acc_create_zero = 8,
  acc_delete = 9,
  acc_attach = 10,
  acc_detach = 11,
  acc_no_create = 12,
  acc_private = 13,
  acc_firstprivate = 14,
  acc_deviceptr = 15,
  acc_getdeviceptr = 16,
};

::std::optional<DataClause> symbolizeDataClause(uint64_t);
::llvm::StringRef stringifyDataClause(DataClause);
::std::optional<DataClause> symbolizeDataClause(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDataClause() {
  return 16;
}


inline ::llvm::StringRef stringifyEnum(DataClause enumValue) {
  return stringifyDataClause(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DataClause> symbolizeEnum<DataClause>(::llvm::StringRef str) {
  return symbolizeDataClause(str);
}

class DataClauseAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DataClause;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DataClauseAttr get(::mlir::MLIRContext *context, DataClause val);
  DataClause getValue() const;
};
} // namespace acc
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::DataClause, ::mlir::acc::DataClause> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::DataClause> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for data clauses supported by OpenACC");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DataClause> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DataClause>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid data clauses supported by OpenACC specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::DataClause value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::DataClause> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::acc::DataClause getEmptyKey() {
    return static_cast<::mlir::acc::DataClause>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::DataClause getTombstoneKey() {
    return static_cast<::mlir::acc::DataClause>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::DataClause &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::acc::DataClause &lhs, const ::mlir::acc::DataClause &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace acc {
// built-in reduction operations supported by OpenACC
enum class ReductionOp : uint32_t {
  redop_add = 0,
  redop_mul = 1,
  redop_max = 2,
  redop_min = 3,
  redop_and = 4,
  redop_or = 5,
  redop_xor = 6,
  redop_leqv = 7,
  redop_lneqv = 8,
  redop_land = 9,
  redop_lor = 10,
};

::std::optional<ReductionOp> symbolizeReductionOp(uint32_t);
::llvm::StringRef stringifyReductionOp(ReductionOp);
::std::optional<ReductionOp> symbolizeReductionOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReductionOp() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(ReductionOp enumValue) {
  return stringifyReductionOp(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReductionOp> symbolizeEnum<ReductionOp>(::llvm::StringRef str) {
  return symbolizeReductionOp(str);
}
} // namespace acc
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::ReductionOp, ::mlir::acc::ReductionOp> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::ReductionOp> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for built-in reduction operations supported by OpenACC");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ReductionOp> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ReductionOp>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid built-in reduction operations supported by OpenACC specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::ReductionOp value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ReductionOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ReductionOp getEmptyKey() {
    return static_cast<::mlir::acc::ReductionOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ReductionOp getTombstoneKey() {
    return static_cast<::mlir::acc::ReductionOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ReductionOp &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ReductionOp &lhs, const ::mlir::acc::ReductionOp &rhs) {
    return lhs == rhs;
  }
};
}

