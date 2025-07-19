#pragma once

#include <ATen/Dispatch.h>

template<typename T> struct XYXY
{
    using value_type = T;
    T x1;
    T y1;
    T x2;
    T y2;
};

template<typename T> struct XYWH
{
    using value_type = T;
    T x;
    T y;
    T w;
    T h;
};

template<typename T> struct CXCYWH
{
    using value_type = T;
    T cx;
    T cy;
    T w;
    T h;
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
