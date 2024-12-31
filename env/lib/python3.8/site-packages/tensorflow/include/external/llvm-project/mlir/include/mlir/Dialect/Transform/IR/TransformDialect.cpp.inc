/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::TransformDialect)
namespace mlir {
namespace transform {

TransformDialect::TransformDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TransformDialect>()) {
  
    getContext()->loadDialect<::mlir::pdl::PDLDialect>();

    getContext()->loadDialect<::mlir::pdl_interp::PDLInterpDialect>();

  initialize();
}

TransformDialect::~TransformDialect() = default;

} // namespace transform
} // namespace mlir
