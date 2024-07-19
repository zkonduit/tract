#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

#define NUM_SIMDGROUP 32

METAL_FUNC uint indices_to_idx_3(uint3 indices, constant const uint strides[3]) {
  return indices.x * strides[2] + indices.y * strides[1] + indices.z * strides[0];
}

template <typename U>
struct MeanOfSquares {
  float simd_reduce(float val, size_t reduce_dim) {
    return simd_sum(val) / static_cast<float>(reduce_dim);
  }

  static constexpr constant float init = 0.0;

  // Operator
  float operator()(float acc, U a) {
    float a_f = static_cast<float>(a);
    return acc + a_f * a_f;
  }
};

template <typename U>
struct Sum {
  U simd_reduce(U val, size_t reduce_dim) {
    return simd_sum(val);
  }

  static constexpr constant U init = U(0);

  // Operator
  U operator()(U acc, U a) {
    return acc + a;
  }
};

template <typename U>
struct Prod {
  U simd_reduce(U val, size_t reduce_dim) {
    return simd_product(val);
  }

  static constexpr constant U init = U(1);

  // Operator
  U operator()(U acc, U a) {
    return acc * a;
  }
};


template<typename F, typename Op>  
[[kernel]] void reduce_nd3(
                device const F *input,
                device F *output,
                constant const size_t input_shape[3], 
                constant const size_t input_strides[3],
                constant const size_t output_strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
        		    uint  tiisg[[thread_index_in_simdgroup]],
        		    uint  tpsg[[threads_per_simdgroup]]
                ) {

    Op op = Op();

    size_t reduce_dim = input_shape[1];

    size_t out_idx = tgpig.x * output_strides[2] 
            + tgpig.y * output_strides[1] 
            + tgpig.z * output_strides[0];

    size_t base_in_idx = tgpig.x * input_strides[2] 
            + tgpig.z * input_strides[0];

    auto partial_acc = Op::init;
    for (size_t i = tiisg; i < reduce_dim; i += tpsg) {
        F el = input[base_in_idx + i * input_strides[1]];
        partial_acc = op(partial_acc, el);
    }
    auto acc = op.simd_reduce(partial_acc, reduce_dim);

    if (tiisg == 0) {
       output[out_idx] = acc;
    }
}

template<typename F>  
[[kernel]] void rms_norm_nd3(
                device const F *input,
                device const F & eps,
                device F *output,
                constant const size_t shape[3], 
                constant const size_t strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
                uint  tiisg[[thread_index_in_simdgroup]],
                uint  tpsg[[threads_per_simdgroup]]
                ) {

    size_t dim = shape[1];

    size_t base_idx = tgpig.x * strides[2] 
            + tgpig.z * strides[0];

    float partial_acc = 0.0;
    for (size_t i = tiisg; i < dim; i += tpsg) {
        F el = input[base_idx + i * strides[1]];
        partial_acc += el * el;
    }
    float mean_of_squares = simd_sum(partial_acc) / static_cast<float>(dim);

    F norm = metal::rsqrt(static_cast<F>(mean_of_squares) + eps);

    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        output[idx] = input[idx] * norm;
    }
}

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

template<typename T>
[[kernel]] void silu(device const T *input[  [buffer(0)]],
                             device T *output [[buffer(1)]],
                             uint tpig[[thread_position_in_grid]]) {
   output[tpig] = Sigmoid()(input[tpig]) * input[tpig];
}


template<typename F>  
[[kernel]] void softmax_nd3(
                device const F *input,
                device F *output,
                constant const size_t shape[3], 
                constant const size_t strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
                uint  tiisg[[thread_index_in_simdgroup]],
                uint  tpsg[[threads_per_simdgroup]]
                ) {

    size_t dim = shape[1];

    size_t base_idx = tgpig.x * strides[2] 
            + tgpig.z * strides[0];

    // Get max value on softmax dim
    float partial_max = -INFINITY;
    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        float el = static_cast<float>(input[idx]);
        partial_max = max(partial_max, el);
    }

    float axis_max = simd_max(partial_max);

    // Compute Sum(exp(x - max))
    float partial_norm = 0;
    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        float el = static_cast<float>(input[idx]);
        float exp_el = fast::exp(el - axis_max);
        partial_norm += exp_el;
        output[idx] = static_cast<F>(exp_el);
    }

    float axis_norm = simd_sum(partial_norm);
    float inv_axis_norm = 1.0 / axis_norm;

    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        float exp_el = static_cast<float>(output[idx]);
        output[idx] = static_cast<F>(exp_el * inv_axis_norm);
    }
}

#define INSTANTIATE_REDUCE(name, op, tname, type)                    \
template [[host_name("nn_ops::reduce_" #name "_nd3_" #tname)]]       \
[[kernel]] void reduce_nd3<type, op<type>>(                          \
        device const type *input,                                    \
        device type *output,                                         \
        constant const size_t input_shape[3],                        \
        constant const size_t input_strides[3],                      \
        constant const size_t output_strides[3],                     \
        uint3  tgpig[[threadgroup_position_in_grid]],                \
        uint  tiisg[[thread_index_in_simdgroup]],                    \
        uint  tpsg[[threads_per_simdgroup]]                          \
    );


INSTANTIATE_REDUCE(mean_of_squares, MeanOfSquares, f32, float)
INSTANTIATE_REDUCE(mean_of_squares, MeanOfSquares, f16, half)
INSTANTIATE_REDUCE(sum, Sum, f32, float)
INSTANTIATE_REDUCE(sum, Sum, f16, half)
INSTANTIATE_REDUCE(prod, Prod, f32, float)
INSTANTIATE_REDUCE(prod, Prod, f16, half)

#define INSTANTIATE_SOFTMAX(tname, type)                             \
template [[host_name("nn_ops::softmax_nd3_" #tname)]]                \
[[kernel]] void softmax_nd3<type>(                                   \
        device const type *input,                                    \
        device type *output,                                         \
        constant const size_t shape[3],                              \
        constant const size_t strides[3],                            \
        uint3  tgpig[[threadgroup_position_in_grid]],                \
        uint  tiisg[[thread_index_in_simdgroup]],                    \
        uint  tpsg[[threads_per_simdgroup]]                          \
    );

INSTANTIATE_SOFTMAX(f32, float)
INSTANTIATE_SOFTMAX(f16, half)

#define INSTANTIATE_RMS_NORM(tname, type)                            \
template [[host_name("nn_ops::rms_norm_nd3_" #tname)]]               \
[[kernel]] void rms_norm_nd3<type>(                                  \
        device const type *input,                                    \
        device const type &eps,                                      \
        device type *output,                                         \
        constant const size_t shape[3],                              \
        constant const size_t strides[3],                            \
        uint3  tgpig[[threadgroup_position_in_grid]],                \
        uint  tiisg[[thread_index_in_simdgroup]],                    \
        uint  tpsg[[threads_per_simdgroup]]                          \
    );

INSTANTIATE_RMS_NORM(f32, float)
INSTANTIATE_RMS_NORM(f16, half)

#define INSTANTIATE_SILU(tname, type)                                   \
template [[host_name("nn_ops::silu_" #tname)]]                          \
[[kernel]] void silu<type>(                                             \
        device const type *input [[buffer(0)]],                         \
        device type *output [[buffer(1)]],                              \
        uint tpig[[thread_position_in_grid]]                            \
    );

INSTANTIATE_SILU(f32, float)
INSTANTIATE_SILU(f16, half)




