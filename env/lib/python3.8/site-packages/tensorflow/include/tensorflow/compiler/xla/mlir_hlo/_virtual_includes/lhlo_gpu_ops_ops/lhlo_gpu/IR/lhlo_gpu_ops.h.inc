/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace lmhlo_gpu {
class AllGatherDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllGatherStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllReduceDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllReduceStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllToAllDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllToAllStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CholeskyOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CollectivePermuteDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CollectivePermuteStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardFilterOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedSideInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CublasLtMatmulF8Op;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CublasLtMatmulOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CudnnConvReorderFilterAndBiasOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CudnnConvReorderFilterOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMMOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ReduceScatterDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ReduceScatterStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllGatherDoneOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllGatherDoneOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllGatherDoneOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AllGatherDoneOpGenericAdaptor : public detail::AllGatherDoneOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllGatherDoneOpGenericAdaptorBase;
public:
  AllGatherDoneOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllGatherDoneOpAdaptor : public AllGatherDoneOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllGatherDoneOpGenericAdaptor::AllGatherDoneOpGenericAdaptor;
  AllGatherDoneOpAdaptor(AllGatherDoneOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllGatherDoneOp : public ::mlir::Op<AllGatherDoneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllGatherDoneOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllGatherDoneOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_gather_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::MutableOperandRange getTokenMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllGatherDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllGatherStartOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllGatherStartOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllGatherStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::IntegerAttr getAllGatherDimensionAttr();
  uint64_t getAllGatherDimension();
};
} // namespace detail
template <typename RangeT>
class AllGatherStartOpGenericAdaptor : public detail::AllGatherStartOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllGatherStartOpGenericAdaptorBase;
public:
  AllGatherStartOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOutputs() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllGatherStartOpAdaptor : public AllGatherStartOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllGatherStartOpGenericAdaptor::AllGatherStartOpGenericAdaptor;
  AllGatherStartOpAdaptor(AllGatherStartOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllGatherStartOp : public ::mlir::Op<AllGatherStartOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllGatherStartOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllGatherStartOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("all_gather_dimension"), ::llvm::StringRef("channel_id"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("is_sync"), ::llvm::StringRef("replica_groups"), ::llvm::StringRef("use_global_device_ids")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAllGatherDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAllGatherDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getChannelIdAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getChannelIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getConstrainLayoutAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getConstrainLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getIsSyncAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getIsSyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getReplicaGroupsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getReplicaGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getUseGlobalDeviceIdsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getUseGlobalDeviceIdsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_gather_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getToken();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::IntegerAttr getAllGatherDimensionAttr();
  uint64_t getAllGatherDimension();
  void setReplicaGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setConstrainLayoutAttr(::mlir::BoolAttr attr);
  void setConstrainLayout(::std::optional<bool> attrValue);
  void setChannelIdAttr(::mlir::mhlo::ChannelHandleAttr attr);
  void setUseGlobalDeviceIdsAttr(::mlir::BoolAttr attr);
  void setUseGlobalDeviceIds(::std::optional<bool> attrValue);
  void setIsSyncAttr(::mlir::BoolAttr attr);
  void setIsSync(bool attrValue);
  void setAllGatherDimensionAttr(::mlir::IntegerAttr attr);
  void setAllGatherDimension(uint64_t attrValue);
  ::mlir::Attribute removeConstrainLayoutAttr();
  ::mlir::Attribute removeChannelIdAttr();
  ::mlir::Attribute removeUseGlobalDeviceIdsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync, ::mlir::IntegerAttr all_gather_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync, ::mlir::IntegerAttr all_gather_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync, uint64_t all_gather_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync, uint64_t all_gather_dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllGatherStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceDoneOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllReduceDoneOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllReduceDoneOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AllReduceDoneOpGenericAdaptor : public detail::AllReduceDoneOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllReduceDoneOpGenericAdaptorBase;
public:
  AllReduceDoneOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllReduceDoneOpAdaptor : public AllReduceDoneOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllReduceDoneOpGenericAdaptor::AllReduceDoneOpGenericAdaptor;
  AllReduceDoneOpAdaptor(AllReduceDoneOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllReduceDoneOp : public ::mlir::Op<AllReduceDoneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceDoneOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllReduceDoneOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::MutableOperandRange getTokenMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceStartOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllReduceStartOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllReduceStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::Region &getComputation();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AllReduceStartOpGenericAdaptor : public detail::AllReduceStartOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllReduceStartOpGenericAdaptorBase;
public:
  AllReduceStartOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOutputs() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllReduceStartOpAdaptor : public AllReduceStartOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllReduceStartOpGenericAdaptor::AllReduceStartOpGenericAdaptor;
  AllReduceStartOpAdaptor(AllReduceStartOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllReduceStartOp : public ::mlir::Op<AllReduceStartOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceStartOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllReduceStartOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("channel_id"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("is_sync"), ::llvm::StringRef("replica_groups"), ::llvm::StringRef("use_global_device_ids")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getChannelIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getChannelIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getConstrainLayoutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getConstrainLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIsSyncAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIsSyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getReplicaGroupsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getReplicaGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getUseGlobalDeviceIdsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getUseGlobalDeviceIdsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getToken();
  ::mlir::Region &getComputation();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  void setReplicaGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setConstrainLayoutAttr(::mlir::BoolAttr attr);
  void setConstrainLayout(::std::optional<bool> attrValue);
  void setChannelIdAttr(::mlir::mhlo::ChannelHandleAttr attr);
  void setUseGlobalDeviceIdsAttr(::mlir::BoolAttr attr);
  void setUseGlobalDeviceIds(::std::optional<bool> attrValue);
  void setIsSyncAttr(::mlir::BoolAttr attr);
  void setIsSync(bool attrValue);
  ::mlir::Attribute removeConstrainLayoutAttr();
  ::mlir::Attribute removeChannelIdAttr();
  ::mlir::Attribute removeUseGlobalDeviceIdsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllToAllDoneOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllToAllDoneOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllToAllDoneOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AllToAllDoneOpGenericAdaptor : public detail::AllToAllDoneOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllToAllDoneOpGenericAdaptorBase;
public:
  AllToAllDoneOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllToAllDoneOpAdaptor : public AllToAllDoneOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllToAllDoneOpGenericAdaptor::AllToAllDoneOpGenericAdaptor;
  AllToAllDoneOpAdaptor(AllToAllDoneOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllToAllDoneOp : public ::mlir::Op<AllToAllDoneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllToAllDoneOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllToAllDoneOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_to_all_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::MutableOperandRange getTokenMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllToAllDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllToAllStartOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllToAllStartOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllToAllStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::IntegerAttr getSplitDimensionAttr();
  ::std::optional<uint64_t> getSplitDimension();
};
} // namespace detail
template <typename RangeT>
class AllToAllStartOpGenericAdaptor : public detail::AllToAllStartOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllToAllStartOpGenericAdaptorBase;
public:
  AllToAllStartOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOutputs() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllToAllStartOpAdaptor : public AllToAllStartOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllToAllStartOpGenericAdaptor::AllToAllStartOpGenericAdaptor;
  AllToAllStartOpAdaptor(AllToAllStartOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllToAllStartOp : public ::mlir::Op<AllToAllStartOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllToAllStartOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllToAllStartOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("channel_id"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("is_sync"), ::llvm::StringRef("replica_groups"), ::llvm::StringRef("split_dimension"), ::llvm::StringRef("use_global_device_ids")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getChannelIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getChannelIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getConstrainLayoutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getConstrainLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIsSyncAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIsSyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getReplicaGroupsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getReplicaGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSplitDimensionAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSplitDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getUseGlobalDeviceIdsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getUseGlobalDeviceIdsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_to_all_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getToken();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::IntegerAttr getSplitDimensionAttr();
  ::std::optional<uint64_t> getSplitDimension();
  void setReplicaGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setConstrainLayoutAttr(::mlir::BoolAttr attr);
  void setConstrainLayout(::std::optional<bool> attrValue);
  void setChannelIdAttr(::mlir::mhlo::ChannelHandleAttr attr);
  void setUseGlobalDeviceIdsAttr(::mlir::BoolAttr attr);
  void setUseGlobalDeviceIds(::std::optional<bool> attrValue);
  void setIsSyncAttr(::mlir::BoolAttr attr);
  void setIsSync(bool attrValue);
  void setSplitDimensionAttr(::mlir::IntegerAttr attr);
  void setSplitDimension(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeConstrainLayoutAttr();
  ::mlir::Attribute removeChannelIdAttr();
  ::mlir::Attribute removeUseGlobalDeviceIdsAttr();
  ::mlir::Attribute removeSplitDimensionAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync, /*optional*/::mlir::IntegerAttr split_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync, /*optional*/::mlir::IntegerAttr split_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync, /*optional*/::mlir::IntegerAttr split_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync, /*optional*/::mlir::IntegerAttr split_dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllToAllStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CholeskyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CholeskyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CholeskyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getIsLowerAttr();
  bool getIsLower();
};
} // namespace detail
template <typename RangeT>
class CholeskyOpGenericAdaptor : public detail::CholeskyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CholeskyOpGenericAdaptorBase;
public:
  CholeskyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(1).begin());
  }

  ValueT getScratch() {
    return (*getODSOperands(2).begin());
  }

  ValueT getInfo() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CholeskyOpAdaptor : public CholeskyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CholeskyOpGenericAdaptor::CholeskyOpGenericAdaptor;
  CholeskyOpAdaptor(CholeskyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CholeskyOp : public ::mlir::Op<CholeskyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CholeskyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CholeskyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_lower")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsLowerAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsLowerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cholesky");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::TypedValue<::mlir::MemRefType> getInfo();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  ::mlir::MutableOperandRange getInfoMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getIsLowerAttr();
  bool getIsLower();
  void setIsLowerAttr(::mlir::BoolAttr attr);
  void setIsLower(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CholeskyOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CollectivePermuteDoneOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CollectivePermuteDoneOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CollectivePermuteDoneOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class CollectivePermuteDoneOpGenericAdaptor : public detail::CollectivePermuteDoneOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CollectivePermuteDoneOpGenericAdaptorBase;
public:
  CollectivePermuteDoneOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CollectivePermuteDoneOpAdaptor : public CollectivePermuteDoneOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CollectivePermuteDoneOpGenericAdaptor::CollectivePermuteDoneOpGenericAdaptor;
  CollectivePermuteDoneOpAdaptor(CollectivePermuteDoneOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CollectivePermuteDoneOp : public ::mlir::Op<CollectivePermuteDoneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CollectivePermuteDoneOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CollectivePermuteDoneOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.collective_permute_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::MutableOperandRange getTokenMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CollectivePermuteDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CollectivePermuteStartOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CollectivePermuteStartOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CollectivePermuteStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getSourceTargetPairsAttr();
  ::mlir::DenseIntElementsAttr getSourceTargetPairs();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
};
} // namespace detail
template <typename RangeT>
class CollectivePermuteStartOpGenericAdaptor : public detail::CollectivePermuteStartOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CollectivePermuteStartOpGenericAdaptorBase;
public:
  CollectivePermuteStartOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getOperand() {
    return (*getODSOperands(0).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CollectivePermuteStartOpAdaptor : public CollectivePermuteStartOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CollectivePermuteStartOpGenericAdaptor::CollectivePermuteStartOpGenericAdaptor;
  CollectivePermuteStartOpAdaptor(CollectivePermuteStartOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CollectivePermuteStartOp : public ::mlir::Op<CollectivePermuteStartOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CollectivePermuteStartOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CollectivePermuteStartOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("channel_id"), ::llvm::StringRef("is_sync"), ::llvm::StringRef("source_target_pairs")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getChannelIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getChannelIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsSyncAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsSyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getSourceTargetPairsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getSourceTargetPairsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.collective_permute_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getOperand();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::MutableOperandRange getOperandMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getToken();
  ::mlir::DenseIntElementsAttr getSourceTargetPairsAttr();
  ::mlir::DenseIntElementsAttr getSourceTargetPairs();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  void setSourceTargetPairsAttr(::mlir::DenseIntElementsAttr attr);
  void setChannelIdAttr(::mlir::mhlo::ChannelHandleAttr attr);
  void setIsSyncAttr(::mlir::BoolAttr attr);
  void setIsSync(bool attrValue);
  ::mlir::Attribute removeChannelIdAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::Value operand, ::mlir::Value output, ::mlir::DenseIntElementsAttr source_target_pairs, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, ::mlir::BoolAttr is_sync);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value output, ::mlir::DenseIntElementsAttr source_target_pairs, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, ::mlir::BoolAttr is_sync);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::Value operand, ::mlir::Value output, ::mlir::DenseIntElementsAttr source_target_pairs, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, bool is_sync);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value output, ::mlir::DenseIntElementsAttr source_target_pairs, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, bool is_sync);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CollectivePermuteStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardFilterOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConvBackwardFilterOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConvBackwardFilterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
};
} // namespace detail
template <typename RangeT>
class ConvBackwardFilterOpGenericAdaptor : public detail::ConvBackwardFilterOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConvBackwardFilterOpGenericAdaptorBase;
public:
  ConvBackwardFilterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDOutput() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDFilter() {
    return (*getODSOperands(2).begin());
  }

  ValueT getScratch() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConvBackwardFilterOpAdaptor : public ConvBackwardFilterOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConvBackwardFilterOpGenericAdaptor::ConvBackwardFilterOpGenericAdaptor;
  ConvBackwardFilterOpAdaptor(ConvBackwardFilterOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConvBackwardFilterOp : public ::mlir::Op<ConvBackwardFilterOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardFilterOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConvBackwardFilterOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardfilter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getDOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getDFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getDOutputMutable();
  ::mlir::MutableOperandRange getDFilterMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setFeatureGroupCount(uint64_t attrValue);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCount(uint64_t attrValue);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setResultScale(::llvm::APFloat attrValue);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindowStridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhsDilationAttr();
  ::mlir::Attribute removeRhsDilationAttr();
  ::mlir::Attribute removeWindowReversalAttr();
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardFilterOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardInputOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConvBackwardInputOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConvBackwardInputOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
};
} // namespace detail
template <typename RangeT>
class ConvBackwardInputOpGenericAdaptor : public detail::ConvBackwardInputOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConvBackwardInputOpGenericAdaptorBase;
public:
  ConvBackwardInputOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDOutput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFilter() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDInput() {
    return (*getODSOperands(2).begin());
  }

  ValueT getScratch() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConvBackwardInputOpAdaptor : public ConvBackwardInputOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConvBackwardInputOpGenericAdaptor::ConvBackwardInputOpGenericAdaptor;
  ConvBackwardInputOpAdaptor(ConvBackwardInputOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConvBackwardInputOp : public ::mlir::Op<ConvBackwardInputOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardInputOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConvBackwardInputOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardinput");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getDOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getDInput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getDOutputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getDInputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setFeatureGroupCount(uint64_t attrValue);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCount(uint64_t attrValue);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setResultScale(::llvm::APFloat attrValue);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindowStridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhsDilationAttr();
  ::mlir::Attribute removeRhsDilationAttr();
  ::mlir::Attribute removeWindowReversalAttr();
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConvForwardFusedOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConvForwardFusedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
};
} // namespace detail
template <typename RangeT>
class ConvForwardFusedOpGenericAdaptor : public detail::ConvForwardFusedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConvForwardFusedOpGenericAdaptorBase;
public:
  ConvForwardFusedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFilter() {
    return (*getODSOperands(1).begin());
  }

  ValueT getBias() {
    return (*getODSOperands(2).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(3).begin());
  }

  ValueT getScratch() {
    return (*getODSOperands(4).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConvForwardFusedOpAdaptor : public ConvForwardFusedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConvForwardFusedOpGenericAdaptor::ConvForwardFusedOpGenericAdaptor;
  ConvForwardFusedOpAdaptor(ConvForwardFusedOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConvForwardFusedOp : public ::mlir::Op<ConvForwardFusedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<5>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConvForwardFusedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("activation_mode"), ::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getActivationModeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getActivationModeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setFeatureGroupCount(uint64_t attrValue);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCount(uint64_t attrValue);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setResultScale(::llvm::APFloat attrValue);
  void setActivationModeAttr(::mlir::lmhlo_gpu::ActivationAttr attr);
  void setActivationMode(::mlir::lmhlo_gpu::Activation attrValue);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindowStridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhsDilationAttr();
  ::mlir::Attribute removeRhsDilationAttr();
  ::mlir::Attribute removeWindowReversalAttr();
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 12 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConvForwardFusedSideInputOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConvForwardFusedSideInputOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::FloatAttr getSideInputScaleAttr();
  ::llvm::APFloat getSideInputScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
};
} // namespace detail
template <typename RangeT>
class ConvForwardFusedSideInputOpGenericAdaptor : public detail::ConvForwardFusedSideInputOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConvForwardFusedSideInputOpGenericAdaptorBase;
public:
  ConvForwardFusedSideInputOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFilter() {
    return (*getODSOperands(1).begin());
  }

  ValueT getBias() {
    return (*getODSOperands(2).begin());
  }

  ValueT getSideInput() {
    return (*getODSOperands(3).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(4).begin());
  }

  ValueT getScratch() {
    return (*getODSOperands(5).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConvForwardFusedSideInputOpAdaptor : public ConvForwardFusedSideInputOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConvForwardFusedSideInputOpGenericAdaptor::ConvForwardFusedSideInputOpGenericAdaptor;
  ConvForwardFusedSideInputOpAdaptor(ConvForwardFusedSideInputOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConvForwardFusedSideInputOp : public ::mlir::Op<ConvForwardFusedSideInputOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedSideInputOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConvForwardFusedSideInputOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("activation_mode"), ::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("side_input_scale"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getActivationModeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getActivationModeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getSideInputScaleAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getSideInputScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(12);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused_with_side_input");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::TypedValue<::mlir::MemRefType> getSideInput();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  ::mlir::MutableOperandRange getSideInputMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::FloatAttr getSideInputScaleAttr();
  ::llvm::APFloat getSideInputScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setFeatureGroupCount(uint64_t attrValue);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCount(uint64_t attrValue);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setResultScale(::llvm::APFloat attrValue);
  void setActivationModeAttr(::mlir::lmhlo_gpu::ActivationAttr attr);
  void setActivationMode(::mlir::lmhlo_gpu::Activation attrValue);
  void setSideInputScaleAttr(::mlir::FloatAttr attr);
  void setSideInputScale(::llvm::APFloat attrValue);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindowStridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhsDilationAttr();
  ::mlir::Attribute removeRhsDilationAttr();
  ::mlir::Attribute removeWindowReversalAttr();
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 13 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConvForwardOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConvForwardOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
};
} // namespace detail
template <typename RangeT>
class ConvForwardOpGenericAdaptor : public detail::ConvForwardOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConvForwardOpGenericAdaptorBase;
public:
  ConvForwardOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFilter() {
    return (*getODSOperands(1).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(2).begin());
  }

  ValueT getScratch() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConvForwardOpAdaptor : public ConvForwardOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConvForwardOpGenericAdaptor::ConvForwardOpGenericAdaptor;
  ConvForwardOpAdaptor(ConvForwardOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConvForwardOp : public ::mlir::Op<ConvForwardOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConvForwardOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::std::optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::std::optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setFeatureGroupCount(uint64_t attrValue);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCount(uint64_t attrValue);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setResultScale(::llvm::APFloat attrValue);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindowStridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhsDilationAttr();
  ::mlir::Attribute removeRhsDilationAttr();
  ::mlir::Attribute removeWindowReversalAttr();
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CublasLtMatmulF8Op declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CublasLtMatmulF8OpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CublasLtMatmulF8OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr getEpilogueAttr();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEpilogue();
  ::mlir::IntegerAttr getAlgorithmAttr();
  uint64_t getAlgorithm();
};
} // namespace detail
template <typename RangeT>
class CublasLtMatmulF8OpGenericAdaptor : public detail::CublasLtMatmulF8OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CublasLtMatmulF8OpGenericAdaptorBase;
public:
  CublasLtMatmulF8OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getC() {
    return (*getODSOperands(2).begin());
  }

  ValueT getAScale() {
    return (*getODSOperands(3).begin());
  }

  ValueT getBScale() {
    return (*getODSOperands(4).begin());
  }

  ValueT getCScale() {
    return (*getODSOperands(5).begin());
  }

  ValueT getDScale() {
    return (*getODSOperands(6).begin());
  }

  ValueT getD() {
    return (*getODSOperands(7).begin());
  }

  ValueT getBias() {
    auto operands = getODSOperands(8);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getDAmax() {
    auto operands = getODSOperands(9);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CublasLtMatmulF8OpAdaptor : public CublasLtMatmulF8OpGenericAdaptor<::mlir::ValueRange> {
public:
  using CublasLtMatmulF8OpGenericAdaptor::CublasLtMatmulF8OpGenericAdaptor;
  CublasLtMatmulF8OpAdaptor(CublasLtMatmulF8Op op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CublasLtMatmulF8Op : public ::mlir::Op<CublasLtMatmulF8Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<8>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CublasLtMatmulF8OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CublasLtMatmulF8OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("algorithm"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("beta"), ::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("epilogue"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("precision_config")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlgorithmAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlgorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getAlphaImagAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getAlphaImagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getAlphaRealAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getAlphaRealAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getBetaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getBetaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getDotDimensionNumbersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getDotDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getEpilogueAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getEpilogueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cublas.lt.matmul.f8");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getA();
  ::mlir::TypedValue<::mlir::MemRefType> getB();
  ::mlir::TypedValue<::mlir::MemRefType> getC();
  ::mlir::TypedValue<::mlir::MemRefType> getAScale();
  ::mlir::TypedValue<::mlir::MemRefType> getBScale();
  ::mlir::TypedValue<::mlir::MemRefType> getCScale();
  ::mlir::TypedValue<::mlir::MemRefType> getDScale();
  ::mlir::TypedValue<::mlir::MemRefType> getD();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::TypedValue<::mlir::MemRefType> getDAmax();
  ::mlir::MutableOperandRange getAMutable();
  ::mlir::MutableOperandRange getBMutable();
  ::mlir::MutableOperandRange getCMutable();
  ::mlir::MutableOperandRange getAScaleMutable();
  ::mlir::MutableOperandRange getBScaleMutable();
  ::mlir::MutableOperandRange getCScaleMutable();
  ::mlir::MutableOperandRange getDScaleMutable();
  ::mlir::MutableOperandRange getDMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  ::mlir::MutableOperandRange getDAmaxMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr getEpilogueAttr();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEpilogue();
  ::mlir::IntegerAttr getAlgorithmAttr();
  uint64_t getAlgorithm();
  void setDotDimensionNumbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setAlphaRealAttr(::mlir::FloatAttr attr);
  void setAlphaReal(::llvm::APFloat attrValue);
  void setAlphaImagAttr(::mlir::FloatAttr attr);
  void setAlphaImag(::llvm::APFloat attrValue);
  void setBetaAttr(::mlir::FloatAttr attr);
  void setBeta(::llvm::APFloat attrValue);
  void setEpilogueAttr(::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr attr);
  void setEpilogue(::mlir::lmhlo_gpu::CublasLtMatmulEpilogue attrValue);
  void setAlgorithmAttr(::mlir::IntegerAttr attr);
  void setAlgorithm(uint64_t attrValue);
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::Value c_scale, ::mlir::Value d_scale, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value d_amax, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr epilogue, ::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::Value c_scale, ::mlir::Value d_scale, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value d_amax, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr epilogue, ::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::Value c_scale, ::mlir::Value d_scale, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value d_amax, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue, uint64_t algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value a_scale, ::mlir::Value b_scale, ::mlir::Value c_scale, ::mlir::Value d_scale, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value d_amax, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue, uint64_t algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CublasLtMatmulF8Op)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CublasLtMatmulOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CublasLtMatmulOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CublasLtMatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr getEpilogueAttr();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEpilogue();
  ::mlir::IntegerAttr getAlgorithmAttr();
  uint64_t getAlgorithm();
};
} // namespace detail
template <typename RangeT>
class CublasLtMatmulOpGenericAdaptor : public detail::CublasLtMatmulOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CublasLtMatmulOpGenericAdaptorBase;
public:
  CublasLtMatmulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getC() {
    return (*getODSOperands(2).begin());
  }

  ValueT getD() {
    return (*getODSOperands(3).begin());
  }

  ValueT getBias() {
    auto operands = getODSOperands(4);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getAux() {
    auto operands = getODSOperands(5);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CublasLtMatmulOpAdaptor : public CublasLtMatmulOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CublasLtMatmulOpGenericAdaptor::CublasLtMatmulOpGenericAdaptor;
  CublasLtMatmulOpAdaptor(CublasLtMatmulOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CublasLtMatmulOp : public ::mlir::Op<CublasLtMatmulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<4>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CublasLtMatmulOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CublasLtMatmulOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("algorithm"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("beta"), ::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("epilogue"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("precision_config")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlgorithmAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlgorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getAlphaImagAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getAlphaImagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getAlphaRealAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getAlphaRealAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getBetaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getBetaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getDotDimensionNumbersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getDotDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getEpilogueAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getEpilogueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cublas.lt.matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getA();
  ::mlir::TypedValue<::mlir::MemRefType> getB();
  ::mlir::TypedValue<::mlir::MemRefType> getC();
  ::mlir::TypedValue<::mlir::MemRefType> getD();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::TypedValue<::mlir::MemRefType> getAux();
  ::mlir::MutableOperandRange getAMutable();
  ::mlir::MutableOperandRange getBMutable();
  ::mlir::MutableOperandRange getCMutable();
  ::mlir::MutableOperandRange getDMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  ::mlir::MutableOperandRange getAuxMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr getEpilogueAttr();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEpilogue();
  ::mlir::IntegerAttr getAlgorithmAttr();
  uint64_t getAlgorithm();
  void setDotDimensionNumbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setAlphaRealAttr(::mlir::FloatAttr attr);
  void setAlphaReal(::llvm::APFloat attrValue);
  void setAlphaImagAttr(::mlir::FloatAttr attr);
  void setAlphaImag(::llvm::APFloat attrValue);
  void setBetaAttr(::mlir::FloatAttr attr);
  void setBeta(::llvm::APFloat attrValue);
  void setEpilogueAttr(::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr attr);
  void setEpilogue(::mlir::lmhlo_gpu::CublasLtMatmulEpilogue attrValue);
  void setAlgorithmAttr(::mlir::IntegerAttr attr);
  void setAlgorithm(uint64_t attrValue);
  ::mlir::Attribute removePrecisionConfigAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value aux, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr epilogue, ::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value aux, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr epilogue, ::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value aux, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue, uint64_t algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, /*optional*/::mlir::Value aux, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue, uint64_t algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CublasLtMatmulOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CudnnConvReorderFilterAndBiasOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CudnnConvReorderFilterAndBiasOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CudnnConvReorderFilterAndBiasOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getFilterDimsAttr();
  ::mlir::DenseIntElementsAttr getFilterDims();
};
} // namespace detail
template <typename RangeT>
class CudnnConvReorderFilterAndBiasOpGenericAdaptor : public detail::CudnnConvReorderFilterAndBiasOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CudnnConvReorderFilterAndBiasOpGenericAdaptorBase;
public:
  CudnnConvReorderFilterAndBiasOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getFilterInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBiasInput() {
    return (*getODSOperands(1).begin());
  }

  ValueT getFilterOutput() {
    return (*getODSOperands(2).begin());
  }

  ValueT getBiasOutput() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CudnnConvReorderFilterAndBiasOpAdaptor : public CudnnConvReorderFilterAndBiasOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CudnnConvReorderFilterAndBiasOpGenericAdaptor::CudnnConvReorderFilterAndBiasOpGenericAdaptor;
  CudnnConvReorderFilterAndBiasOpAdaptor(CudnnConvReorderFilterAndBiasOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CudnnConvReorderFilterAndBiasOp : public ::mlir::Op<CudnnConvReorderFilterAndBiasOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CudnnConvReorderFilterAndBiasOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CudnnConvReorderFilterAndBiasOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("filter_dims")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getFilterDimsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFilterDimsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cudnn_conv_reorder_filter_and_bias");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getFilterInput();
  ::mlir::TypedValue<::mlir::MemRefType> getBiasInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilterOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getBiasOutput();
  ::mlir::MutableOperandRange getFilterInputMutable();
  ::mlir::MutableOperandRange getBiasInputMutable();
  ::mlir::MutableOperandRange getFilterOutputMutable();
  ::mlir::MutableOperandRange getBiasOutputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getFilterDimsAttr();
  ::mlir::DenseIntElementsAttr getFilterDims();
  void setFilterDimsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter_input, ::mlir::Value bias_input, ::mlir::Value filter_output, ::mlir::Value bias_output, ::mlir::DenseIntElementsAttr filter_dims);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter_input, ::mlir::Value bias_input, ::mlir::Value filter_output, ::mlir::Value bias_output, ::mlir::DenseIntElementsAttr filter_dims);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CudnnConvReorderFilterAndBiasOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CudnnConvReorderFilterOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CudnnConvReorderFilterOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CudnnConvReorderFilterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getFilterDimsAttr();
  ::mlir::DenseIntElementsAttr getFilterDims();
};
} // namespace detail
template <typename RangeT>
class CudnnConvReorderFilterOpGenericAdaptor : public detail::CudnnConvReorderFilterOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CudnnConvReorderFilterOpGenericAdaptorBase;
public:
  CudnnConvReorderFilterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getFilterInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFilterOutput() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CudnnConvReorderFilterOpAdaptor : public CudnnConvReorderFilterOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CudnnConvReorderFilterOpGenericAdaptor::CudnnConvReorderFilterOpGenericAdaptor;
  CudnnConvReorderFilterOpAdaptor(CudnnConvReorderFilterOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CudnnConvReorderFilterOp : public ::mlir::Op<CudnnConvReorderFilterOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CudnnConvReorderFilterOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CudnnConvReorderFilterOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("filter_dims")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getFilterDimsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFilterDimsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cudnn_conv_reorder_filter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getFilterInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilterOutput();
  ::mlir::MutableOperandRange getFilterInputMutable();
  ::mlir::MutableOperandRange getFilterOutputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getFilterDimsAttr();
  ::mlir::DenseIntElementsAttr getFilterDims();
  void setFilterDimsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter_input, ::mlir::Value filter_output, ::mlir::DenseIntElementsAttr filter_dims);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter_input, ::mlir::Value filter_output, ::mlir::DenseIntElementsAttr filter_dims);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CudnnConvReorderFilterOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMMOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GEMMOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GEMMOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::IntegerAttr getAlgorithmAttr();
  ::std::optional<uint64_t> getAlgorithm();
};
} // namespace detail
template <typename RangeT>
class GEMMOpGenericAdaptor : public detail::GEMMOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GEMMOpGenericAdaptorBase;
public:
  GEMMOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getC() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GEMMOpAdaptor : public GEMMOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GEMMOpGenericAdaptor::GEMMOpGenericAdaptor;
  GEMMOpAdaptor(GEMMOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GEMMOp : public ::mlir::Op<GEMMOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMMOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GEMMOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("algorithm"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("beta"), ::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("precision_config")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlgorithmAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlgorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getAlphaImagAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getAlphaImagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getAlphaRealAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getAlphaRealAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getBetaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getBetaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getDotDimensionNumbersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getDotDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.gemm");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getA();
  ::mlir::TypedValue<::mlir::MemRefType> getB();
  ::mlir::TypedValue<::mlir::MemRefType> getC();
  ::mlir::MutableOperandRange getAMutable();
  ::mlir::MutableOperandRange getBMutable();
  ::mlir::MutableOperandRange getCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::std::optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::IntegerAttr getAlgorithmAttr();
  ::std::optional<uint64_t> getAlgorithm();
  void setDotDimensionNumbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setAlphaRealAttr(::mlir::FloatAttr attr);
  void setAlphaReal(::llvm::APFloat attrValue);
  void setAlphaImagAttr(::mlir::FloatAttr attr);
  void setAlphaImag(::llvm::APFloat attrValue);
  void setBetaAttr(::mlir::FloatAttr attr);
  void setBeta(::llvm::APFloat attrValue);
  void setAlgorithmAttr(::mlir::IntegerAttr attr);
  void setAlgorithm(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removePrecisionConfigAttr();
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::GEMMOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ReduceScatterDoneOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReduceScatterDoneOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReduceScatterDoneOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReduceScatterDoneOpGenericAdaptor : public detail::ReduceScatterDoneOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReduceScatterDoneOpGenericAdaptorBase;
public:
  ReduceScatterDoneOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReduceScatterDoneOpAdaptor : public ReduceScatterDoneOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReduceScatterDoneOpGenericAdaptor::ReduceScatterDoneOpGenericAdaptor;
  ReduceScatterDoneOpAdaptor(ReduceScatterDoneOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReduceScatterDoneOp : public ::mlir::Op<ReduceScatterDoneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduceScatterDoneOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReduceScatterDoneOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.reduce_scatter_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::MutableOperandRange getTokenMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ReduceScatterDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ReduceScatterStartOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReduceScatterStartOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReduceScatterStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::IntegerAttr getScatterDimensionAttr();
  uint64_t getScatterDimension();
  ::mlir::Region &getComputation();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ReduceScatterStartOpGenericAdaptor : public detail::ReduceScatterStartOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReduceScatterStartOpGenericAdaptorBase;
public:
  ReduceScatterStartOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOutputs() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReduceScatterStartOpAdaptor : public ReduceScatterStartOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReduceScatterStartOpGenericAdaptor::ReduceScatterStartOpGenericAdaptor;
  ReduceScatterStartOpAdaptor(ReduceScatterStartOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReduceScatterStartOp : public ::mlir::Op<ReduceScatterStartOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduceScatterStartOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReduceScatterStartOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("channel_id"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("is_sync"), ::llvm::StringRef("replica_groups"), ::llvm::StringRef("scatter_dimension"), ::llvm::StringRef("use_global_device_ids")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getChannelIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getChannelIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getConstrainLayoutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getConstrainLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIsSyncAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIsSyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getReplicaGroupsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getReplicaGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getScatterDimensionAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getScatterDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getUseGlobalDeviceIdsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getUseGlobalDeviceIdsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.reduce_scatter_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getToken();
  ::mlir::Region &getComputation();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::std::optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::BoolAttr getIsSyncAttr();
  bool getIsSync();
  ::mlir::IntegerAttr getScatterDimensionAttr();
  uint64_t getScatterDimension();
  void setReplicaGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setConstrainLayoutAttr(::mlir::BoolAttr attr);
  void setConstrainLayout(::std::optional<bool> attrValue);
  void setChannelIdAttr(::mlir::mhlo::ChannelHandleAttr attr);
  void setUseGlobalDeviceIdsAttr(::mlir::BoolAttr attr);
  void setUseGlobalDeviceIds(::std::optional<bool> attrValue);
  void setIsSyncAttr(::mlir::BoolAttr attr);
  void setIsSync(bool attrValue);
  void setScatterDimensionAttr(::mlir::IntegerAttr attr);
  void setScatterDimension(uint64_t attrValue);
  ::mlir::Attribute removeConstrainLayoutAttr();
  ::mlir::Attribute removeChannelIdAttr();
  ::mlir::Attribute removeUseGlobalDeviceIdsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync, ::mlir::IntegerAttr scatter_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids, ::mlir::BoolAttr is_sync, ::mlir::IntegerAttr scatter_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync, uint64_t scatter_dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids, bool is_sync, uint64_t scatter_dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ReduceScatterStartOp)


#endif  // GET_OP_CLASSES

