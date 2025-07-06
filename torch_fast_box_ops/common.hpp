#pragma once

template<typename T> struct XYXY
{
    T x1;
    T y1;
    T x2;
    T y2;
};

template<typename T> struct XYWH
{
    T x;
    T y;
    T w;
    T h;
};

template<typename T> struct CXCYWH
{
    T cx;
    T cy;
    T w;
    T h;
};
