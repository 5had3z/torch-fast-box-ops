#pragma once

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

#ifdef __CUDACC__
#define FN_QUAL __host__ __device__
#else
#define FN_QUAL
#endif

template<typename T> FN_QUAL CXCYWH<T> convert_box(const XYXY<T> box, xyxy_tag, cxcywh_tag)
{
    CXCYWH<T> result;
    result.cx = (box.x1 + box.x2) * 0.5f;
    result.cy = (box.y1 + box.y2) * 0.5f;
    result.w = box.x2 - box.x1;
    result.h = box.y2 - box.y1;
    return result;
}

template<typename T> FN_QUAL CXCYWH<T> convert_box(const XYWH<T> box, xywh_tag, cxcywh_tag)
{
    CXCYWH<T> result;
    result.cx = box.x + box.w * 0.5f;
    result.cy = box.y + box.h * 0.5f;
    result.w = box.w;
    result.h = box.h;
    return result;
}

template<typename T> FN_QUAL XYXY<T> convert_box(const CXCYWH<T> box, cxcywh_tag, xyxy_tag)
{
    XYXY<T> result;
    result.x1 = box.cx - box.w * 0.5f;
    result.y1 = box.cy - box.h * 0.5f;
    result.x2 = box.cx + box.w * 0.5f;
    result.y2 = box.cy + box.h * 0.5f;
    return result;
}

template<typename T> FN_QUAL XYXY<T> convert_box(const XYWH<T> box, xywh_tag, xyxy_tag)
{
    XYXY<T> result;
    result.x1 = box.x;
    result.y1 = box.y;
    result.x2 = box.x + box.w;
    result.y2 = box.y + box.h;
    return result;
}

template<typename T> FN_QUAL XYWH<T> convert_box(const CXCYWH<T> box, cxcywh_tag, xywh_tag)
{
    XYWH<T> result;
    result.x = box.cx - box.w * 0.5f;
    result.y = box.cy - box.h * 0.5f;
    result.w = box.w;
    result.h = box.h;
    return result;
}

template<typename T> FN_QUAL XYWH<T> convert_box(const XYXY<T> box, xyxy_tag, xywh_tag)
{
    XYWH<T> result;
    result.x = box.x1;
    result.y = box.y1;
    result.w = box.x2 - box.x1;
    result.h = box.y2 - box.y1;
    return result;
}

#undef FN_QUAL
