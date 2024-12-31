/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {
// Activation applied with fused convolution
enum class Activation : uint32_t {
  None = 0,
  Sigmoid = 1,
  Tanh = 2,
  Relu = 3,
  Relu6 = 4,
  ReluX = 5,
  BandPass = 6,
  Elu = 7,
};

::std::optional<Activation> symbolizeActivation(uint32_t);
::llvm::StringRef stringifyActivation(Activation);
::std::optional<Activation> symbolizeActivation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForActivation() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(Activation enumValue) {
  return stringifyActivation(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Activation> symbolizeEnum<Activation>(::llvm::StringRef str) {
  return symbolizeActivation(str);
}
} // namespace lmhlo_gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::lmhlo_gpu::Activation, ::mlir::lmhlo_gpu::Activation> {
  template <typename ParserT>
  static FailureOr<::mlir::lmhlo_gpu::Activation> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Activation applied with fused convolution");

    // Symbolize the keyword.
    if (::std::optional<::mlir::lmhlo_gpu::Activation> attr = ::mlir::lmhlo_gpu::symbolizeEnum<::mlir::lmhlo_gpu::Activation>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Activation applied with fused convolution specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::lmhlo_gpu::Activation value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::lmhlo_gpu::Activation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::lmhlo_gpu::Activation getEmptyKey() {
    return static_cast<::mlir::lmhlo_gpu::Activation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::lmhlo_gpu::Activation getTombstoneKey() {
    return static_cast<::mlir::lmhlo_gpu::Activation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::lmhlo_gpu::Activation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::lmhlo_gpu::Activation &lhs, const ::mlir::lmhlo_gpu::Activation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace lmhlo_gpu {
// Epilogue for cublasLt matmul
enum class CublasLtMatmulEpilogue : uint32_t {
  Default = 0,
  Bias = 1,
  Relu = 2,
  BiasRelu = 3,
  Gelu = 4,
  BiasGelu = 5,
  GeluAux = 6,
  BiasGeluAux = 7,
};

::std::optional<CublasLtMatmulEpilogue> symbolizeCublasLtMatmulEpilogue(uint32_t);
::llvm::StringRef stringifyCublasLtMatmulEpilogue(CublasLtMatmulEpilogue);
::std::optional<CublasLtMatmulEpilogue> symbolizeCublasLtMatmulEpilogue(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCublasLtMatmulEpilogue() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(CublasLtMatmulEpilogue enumValue) {
  return stringifyCublasLtMatmulEpilogue(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CublasLtMatmulEpilogue> symbolizeEnum<CublasLtMatmulEpilogue>(::llvm::StringRef str) {
  return symbolizeCublasLtMatmulEpilogue(str);
}
} // namespace lmhlo_gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue> {
  template <typename ParserT>
  static FailureOr<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Epilogue for cublasLt matmul");

    // Symbolize the keyword.
    if (::std::optional<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue> attr = ::mlir::lmhlo_gpu::symbolizeEnum<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Epilogue for cublasLt matmul specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEmptyKey() {
    return static_cast<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getTombstoneKey() {
    return static_cast<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue &lhs, const ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue &rhs) {
    return lhs == rhs;
  }
};
}

