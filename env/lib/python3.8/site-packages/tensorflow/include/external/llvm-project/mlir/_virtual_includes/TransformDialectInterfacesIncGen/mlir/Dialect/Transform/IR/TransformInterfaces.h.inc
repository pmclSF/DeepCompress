/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
class TransformOpInterface;
namespace detail {
struct TransformOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DiagnosedSilenceableFailure (*apply)(const Concept *impl, ::mlir::Operation *, ::mlir::transform::TransformResults &, ::mlir::transform::TransformState &);
    bool (*allowsRepeatedHandleOperands)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    Model() : Concept{apply, allowsRepeatedHandleOperands} {}

    static inline ::mlir::DiagnosedSilenceableFailure apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
    static inline bool allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    FallbackModel() : Concept{apply, allowsRepeatedHandleOperands} {}

    static inline ::mlir::DiagnosedSilenceableFailure apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
    static inline bool allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool allowsRepeatedHandleOperands(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct TransformOpInterfaceTrait;

} // namespace detail
class TransformOpInterface : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TransformOpInterfaceTrait<ConcreteOp> {};
  /// Applies the transformation represented by the current operation. This
  /// accepts as arguments the object that must be populated with results of
  /// the current transformation and a transformation state object that can be
  /// used for queries, e.g., to obtain the list of operations on which the
  /// transformation represented by the current op is targeted. Returns a
  /// special status object indicating whether the transformation succeeded
  /// or failed, and, if it failed, whether the failure is recoverable or not.
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  /// Indicates whether the op instance allows its handle operands to be
  /// associated with the same payload operations.
  bool allowsRepeatedHandleOperands();

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation. Silenceable failure must be suppressed or reported
    /// explicitly at some later time.
    DiagnosedSilenceableFailure
    emitSilenceableError(const ::llvm::Twine &message = {}) {
      return ::mlir::emitSilenceableFailure((*this), message);
    }

    /// Creates the definite failure object with a diagnostic located at the
    /// current operation. Definite failure will be reported when the object
    /// is destroyed or converted.
    DiagnosedDefiniteFailure
    emitDefiniteFailure(const ::llvm::Twine &message = {}) {
      return ::mlir::emitDefiniteFailure((*this), message);
    }

    /// Emits a generic definite failure for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
        ::mlir::Operation *target) {
      auto diag = ::mlir::emitDefiniteFailure((*this), "failed to apply");
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure((*this)->getLoc());
      diag << (*this)->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return diag;
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct TransformOpInterfaceTrait : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Indicates whether the op instance allows its handle operands to be
    /// associated with the same payload operations.
    bool allowsRepeatedHandleOperands() {
      return false;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::transform::detail::verifyTransformOpInterface(op);
    }

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation. Silenceable failure must be suppressed or reported
    /// explicitly at some later time.
    DiagnosedSilenceableFailure
    emitSilenceableError(const ::llvm::Twine &message = {}) {
      return ::mlir::emitSilenceableFailure((*static_cast<ConcreteOp *>(this)), message);
    }

    /// Creates the definite failure object with a diagnostic located at the
    /// current operation. Definite failure will be reported when the object
    /// is destroyed or converted.
    DiagnosedDefiniteFailure
    emitDefiniteFailure(const ::llvm::Twine &message = {}) {
      return ::mlir::emitDefiniteFailure((*static_cast<ConcreteOp *>(this)), message);
    }

    /// Emits a generic definite failure for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
        ::mlir::Operation *target) {
      auto diag = ::mlir::emitDefiniteFailure((*static_cast<ConcreteOp *>(this)), "failed to apply");
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure((*static_cast<ConcreteOp *>(this))->getLoc());
      diag << (*static_cast<ConcreteOp *>(this))->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return diag;
    }
  
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
template<typename ConcreteOp>
::mlir::DiagnosedSilenceableFailure detail::TransformOpInterfaceInterfaceTraits::Model<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).apply(transformResults, state);
}
template<typename ConcreteOp>
bool detail::TransformOpInterfaceInterfaceTraits::Model<ConcreteOp>::allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).allowsRepeatedHandleOperands();
}
template<typename ConcreteOp>
::mlir::DiagnosedSilenceableFailure detail::TransformOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return static_cast<const ConcreteOp *>(impl)->apply(tablegen_opaque_val, transformResults, state);
}
template<typename ConcreteOp>
bool detail::TransformOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->allowsRepeatedHandleOperands(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::TransformOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::allowsRepeatedHandleOperands(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
} // namespace transform
} // namespace mlir
