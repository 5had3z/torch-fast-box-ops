#pragma once

#include <ATen/Dispatch.h>

#include "kernel.cuh"

template<typename T> struct box_aligned_size
{
    static constexpr size_t value = sizeof(T) * 4;
};

template<typename T, int N> struct aligned_type;

template<> struct aligned_type<float, 4>
{
    using vec_t = float4;
};

template<> struct aligned_type<int32_t, 4>
{
    using vec_t = int4;
};

template<> struct aligned_type<c10::Half, 4>
{
    using vec_t = ushort4;// for half4
};

template<> struct aligned_type<c10::BFloat16, 4>
{
    using vec_t = ushort4;// for bfloat16
};

template<> struct aligned_type<double, 4>
{
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
    using vec_t = double4_32a;// CUDA 13+ uses explicit alignment types
#else
    using vec_t = double4;
#endif
};

template<typename T> struct alignas(box_aligned_size<T>::value) XYXY
{
    using value_type = T;
    union {
        struct
        {
            T x1, y1, x2, y2;
        };
        typename aligned_type<T, 4>::vec_t vec;// maps to float4, int4, etc.
    };

    [[nodiscard]] TFBO_HOST_DEVICE auto width() const noexcept -> T { return x2 - x1; }

    [[nodiscard]] TFBO_HOST_DEVICE auto height() const noexcept -> T { return y2 - y1; }
};

template<typename T> struct alignas(box_aligned_size<T>::value) XYWH
{
    using value_type = T;
    union {
        struct
        {
            T x, y, w, h;
        };
        typename aligned_type<T, 4>::vec_t vec;// maps to float4, int4, etc.
    };
};

template<typename T> struct alignas(box_aligned_size<T>::value) CXCYWH
{
    using value_type = T;
    union {
        struct
        {
            T cx, cy, w, h;
        };
        typename aligned_type<T, 4>::vec_t vec;// maps to float4, int4, etc.
    };
};


struct box_tag
{
};
struct xyxy_tag : box_tag
{
};
struct xywh_tag : box_tag
{
};
struct cxcywh_tag : box_tag
{
};

template<template<typename> typename BoxType> struct box_tag_map;

template<> struct box_tag_map<XYXY>
{
    using type = xyxy_tag;
};
template<> struct box_tag_map<XYWH>
{
    using type = xywh_tag;
};
template<> struct box_tag_map<CXCYWH>
{
    using type = cxcywh_tag;
};

#define TFBO_DISPATCH_CASE_BOX_TYPES(...)                   \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)   \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)

#define TFBO_DISPATCH_BOX_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, TFBO_DISPATCH_CASE_BOX_TYPES(__VA_ARGS__))
