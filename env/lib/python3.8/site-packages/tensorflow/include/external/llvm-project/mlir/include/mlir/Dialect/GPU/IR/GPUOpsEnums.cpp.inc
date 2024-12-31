/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyBlocks(Blocks val) {
  switch (val) {
    case Blocks::DimX: return "x";
    case Blocks::DimY: return "y";
    case Blocks::DimZ: return "z";
  }
  return "";
}

::std::optional<Blocks> symbolizeBlocks(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Blocks>>(str)
      .Case("x", Blocks::DimX)
      .Case("y", Blocks::DimY)
      .Case("z", Blocks::DimZ)
      .Default(::std::nullopt);
}
::std::optional<Blocks> symbolizeBlocks(uint64_t value) {
  switch (value) {
  case 0: return Blocks::DimX;
  case 1: return Blocks::DimY;
  case 2: return Blocks::DimZ;
  default: return ::std::nullopt;
  }
}

bool BlocksAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
BlocksAttr BlocksAttr::get(::mlir::MLIRContext *context, Blocks val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<BlocksAttr>();
}
Blocks BlocksAttr::getValue() const {
  return static_cast<Blocks>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyAddressSpace(AddressSpace val) {
  switch (val) {
    case AddressSpace::Global: return "global";
    case AddressSpace::Workgroup: return "workgroup";
    case AddressSpace::Private: return "private";
  }
  return "";
}

::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AddressSpace>>(str)
      .Case("global", AddressSpace::Global)
      .Case("workgroup", AddressSpace::Workgroup)
      .Case("private", AddressSpace::Private)
      .Default(::std::nullopt);
}
::std::optional<AddressSpace> symbolizeAddressSpace(uint32_t value) {
  switch (value) {
  case 1: return AddressSpace::Global;
  case 2: return AddressSpace::Workgroup;
  case 3: return AddressSpace::Private;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyAllReduceOperation(AllReduceOperation val) {
  switch (val) {
    case AllReduceOperation::ADD: return "add";
    case AllReduceOperation::AND: return "and";
    case AllReduceOperation::MAX: return "max";
    case AllReduceOperation::MIN: return "min";
    case AllReduceOperation::MUL: return "mul";
    case AllReduceOperation::OR: return "or";
    case AllReduceOperation::XOR: return "xor";
  }
  return "";
}

::std::optional<AllReduceOperation> symbolizeAllReduceOperation(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AllReduceOperation>>(str)
      .Case("add", AllReduceOperation::ADD)
      .Case("and", AllReduceOperation::AND)
      .Case("max", AllReduceOperation::MAX)
      .Case("min", AllReduceOperation::MIN)
      .Case("mul", AllReduceOperation::MUL)
      .Case("or", AllReduceOperation::OR)
      .Case("xor", AllReduceOperation::XOR)
      .Default(::std::nullopt);
}
::std::optional<AllReduceOperation> symbolizeAllReduceOperation(uint32_t value) {
  switch (value) {
  case 0: return AllReduceOperation::ADD;
  case 1: return AllReduceOperation::AND;
  case 2: return AllReduceOperation::MAX;
  case 3: return AllReduceOperation::MIN;
  case 4: return AllReduceOperation::MUL;
  case 5: return AllReduceOperation::OR;
  case 6: return AllReduceOperation::XOR;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyDimension(Dimension val) {
  switch (val) {
    case Dimension::x: return "x";
    case Dimension::y: return "y";
    case Dimension::z: return "z";
  }
  return "";
}

::std::optional<Dimension> symbolizeDimension(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Dimension>>(str)
      .Case("x", Dimension::x)
      .Case("y", Dimension::y)
      .Case("z", Dimension::z)
      .Default(::std::nullopt);
}
::std::optional<Dimension> symbolizeDimension(uint32_t value) {
  switch (value) {
  case 0: return Dimension::x;
  case 1: return Dimension::y;
  case 2: return Dimension::z;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyShuffleMode(ShuffleMode val) {
  switch (val) {
    case ShuffleMode::XOR: return "xor";
    case ShuffleMode::UP: return "up";
    case ShuffleMode::DOWN: return "down";
    case ShuffleMode::IDX: return "idx";
  }
  return "";
}

::std::optional<ShuffleMode> symbolizeShuffleMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ShuffleMode>>(str)
      .Case("xor", ShuffleMode::XOR)
      .Case("up", ShuffleMode::UP)
      .Case("down", ShuffleMode::DOWN)
      .Case("idx", ShuffleMode::IDX)
      .Default(::std::nullopt);
}
::std::optional<ShuffleMode> symbolizeShuffleMode(uint32_t value) {
  switch (value) {
  case 0: return ShuffleMode::XOR;
  case 2: return ShuffleMode::UP;
  case 1: return ShuffleMode::DOWN;
  case 3: return ShuffleMode::IDX;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyLinearId(LinearId val) {
  switch (val) {
    case LinearId::DimX: return "x";
    case LinearId::DimY: return "y";
    case LinearId::DimZ: return "z";
  }
  return "";
}

::std::optional<LinearId> symbolizeLinearId(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<LinearId>>(str)
      .Case("x", LinearId::DimX)
      .Case("y", LinearId::DimY)
      .Case("z", LinearId::DimZ)
      .Default(::std::nullopt);
}
::std::optional<LinearId> symbolizeLinearId(uint64_t value) {
  switch (value) {
  case 0: return LinearId::DimX;
  case 1: return LinearId::DimY;
  case 2: return LinearId::DimZ;
  default: return ::std::nullopt;
  }
}

bool LinearIdAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
LinearIdAttr LinearIdAttr::get(::mlir::MLIRContext *context, LinearId val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<LinearIdAttr>();
}
LinearId LinearIdAttr::getValue() const {
  return static_cast<LinearId>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyMMAElementwiseOp(MMAElementwiseOp val) {
  switch (val) {
    case MMAElementwiseOp::ADDF: return "addf";
    case MMAElementwiseOp::MULF: return "mulf";
    case MMAElementwiseOp::SUBF: return "subf";
    case MMAElementwiseOp::MAXF: return "maxf";
    case MMAElementwiseOp::MINF: return "minf";
    case MMAElementwiseOp::DIVF: return "divf";
    case MMAElementwiseOp::ADDI: return "addi";
    case MMAElementwiseOp::MULI: return "muli";
    case MMAElementwiseOp::SUBI: return "subi";
    case MMAElementwiseOp::DIVS: return "divs";
    case MMAElementwiseOp::DIVU: return "divu";
    case MMAElementwiseOp::NEGATEF: return "negatef";
    case MMAElementwiseOp::NEGATES: return "negates";
  }
  return "";
}

::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MMAElementwiseOp>>(str)
      .Case("addf", MMAElementwiseOp::ADDF)
      .Case("mulf", MMAElementwiseOp::MULF)
      .Case("subf", MMAElementwiseOp::SUBF)
      .Case("maxf", MMAElementwiseOp::MAXF)
      .Case("minf", MMAElementwiseOp::MINF)
      .Case("divf", MMAElementwiseOp::DIVF)
      .Case("addi", MMAElementwiseOp::ADDI)
      .Case("muli", MMAElementwiseOp::MULI)
      .Case("subi", MMAElementwiseOp::SUBI)
      .Case("divs", MMAElementwiseOp::DIVS)
      .Case("divu", MMAElementwiseOp::DIVU)
      .Case("negatef", MMAElementwiseOp::NEGATEF)
      .Case("negates", MMAElementwiseOp::NEGATES)
      .Default(::std::nullopt);
}
::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(uint32_t value) {
  switch (value) {
  case 0: return MMAElementwiseOp::ADDF;
  case 1: return MMAElementwiseOp::MULF;
  case 2: return MMAElementwiseOp::SUBF;
  case 3: return MMAElementwiseOp::MAXF;
  case 4: return MMAElementwiseOp::MINF;
  case 5: return MMAElementwiseOp::DIVF;
  case 6: return MMAElementwiseOp::ADDI;
  case 7: return MMAElementwiseOp::MULI;
  case 8: return MMAElementwiseOp::SUBI;
  case 9: return MMAElementwiseOp::DIVS;
  case 10: return MMAElementwiseOp::DIVU;
  case 11: return MMAElementwiseOp::NEGATEF;
  case 12: return MMAElementwiseOp::NEGATES;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyProcessor(Processor val) {
  switch (val) {
    case Processor::BlockX: return "block_x";
    case Processor::BlockY: return "block_y";
    case Processor::BlockZ: return "block_z";
    case Processor::ThreadX: return "thread_x";
    case Processor::ThreadY: return "thread_y";
    case Processor::ThreadZ: return "thread_z";
    case Processor::Sequential: return "sequential";
  }
  return "";
}

::std::optional<Processor> symbolizeProcessor(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Processor>>(str)
      .Case("block_x", Processor::BlockX)
      .Case("block_y", Processor::BlockY)
      .Case("block_z", Processor::BlockZ)
      .Case("thread_x", Processor::ThreadX)
      .Case("thread_y", Processor::ThreadY)
      .Case("thread_z", Processor::ThreadZ)
      .Case("sequential", Processor::Sequential)
      .Default(::std::nullopt);
}
::std::optional<Processor> symbolizeProcessor(uint64_t value) {
  switch (value) {
  case 0: return Processor::BlockX;
  case 1: return Processor::BlockY;
  case 2: return Processor::BlockZ;
  case 3: return Processor::ThreadX;
  case 4: return Processor::ThreadY;
  case 5: return Processor::ThreadZ;
  case 6: return Processor::Sequential;
  default: return ::std::nullopt;
  }
}

bool ProcessorAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)));
}
ProcessorAttr ProcessorAttr::get(::mlir::MLIRContext *context, Processor val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<ProcessorAttr>();
}
Processor ProcessorAttr::getValue() const {
  return static_cast<Processor>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyThreads(Threads val) {
  switch (val) {
    case Threads::DimX: return "x";
    case Threads::DimY: return "y";
    case Threads::DimZ: return "z";
  }
  return "";
}

::std::optional<Threads> symbolizeThreads(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Threads>>(str)
      .Case("x", Threads::DimX)
      .Case("y", Threads::DimY)
      .Case("z", Threads::DimZ)
      .Default(::std::nullopt);
}
::std::optional<Threads> symbolizeThreads(uint64_t value) {
  switch (value) {
  case 0: return Threads::DimX;
  case 1: return Threads::DimY;
  case 2: return Threads::DimZ;
  default: return ::std::nullopt;
  }
}

bool ThreadsAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
ThreadsAttr ThreadsAttr::get(::mlir::MLIRContext *context, Threads val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<ThreadsAttr>();
}
Threads ThreadsAttr::getValue() const {
  return static_cast<Threads>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyWarps(Warps val) {
  switch (val) {
    case Warps::DimX: return "x";
    case Warps::DimY: return "y";
    case Warps::DimZ: return "z";
  }
  return "";
}

::std::optional<Warps> symbolizeWarps(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Warps>>(str)
      .Case("x", Warps::DimX)
      .Case("y", Warps::DimY)
      .Case("z", Warps::DimZ)
      .Default(::std::nullopt);
}
::std::optional<Warps> symbolizeWarps(uint64_t value) {
  switch (value) {
  case 0: return Warps::DimX;
  case 1: return Warps::DimY;
  case 2: return Warps::DimZ;
  default: return ::std::nullopt;
  }
}

bool WarpsAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
WarpsAttr WarpsAttr::get(::mlir::MLIRContext *context, Warps val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<WarpsAttr>();
}
Warps WarpsAttr::getValue() const {
  return static_cast<Warps>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

