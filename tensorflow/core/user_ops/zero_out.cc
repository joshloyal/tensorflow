#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: T")
    .Output("zeroed: T")
    .Attr("T: realnumbertype = DT_INT32")
    .Attr("preserve_index: int = 0");


template <typename T>
class ZeroOutOp: public OpKernel {
    public:
        explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
            // Get the index of the value preserve
            OP_REQUIRES_OK(context,
                           context->GetAttr("preserve_index", &preserve_index_));

            // Check that preserve_index is positive
            OP_REQUIRES(context, preserve_index_ >= 0,
                        errors::InvalidArgument("Need preserve_index >= 0, got ",
                        preserve_index_));
            }
        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);

            OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                        errors::InvalidArgument("ZeroOut expects a 1-D vector."));

            auto input = input_tensor.flat<T>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                             &output_tensor));
            auto output = output_tensor->template flat<T>();

            // Check the preserve_index is in range
            OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                        errors::InvalidArgument("preserve_index out of range"));

            // Set all but the first element of the output tensor to 0
            const int N = input.size();
            for (int i = 1; i < N; i++) {
                output(i) = input(i) * input(i);
            }

            // Preserve the first input value if possible
            output(preserve_index_) = input(preserve_index_);
        }
    private:
        int preserve_index_;
};

#define REGISTER_KERNEL(type)                                           \
    REGISTER_KERNEL_BUILDER(                                            \
        Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
        ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
