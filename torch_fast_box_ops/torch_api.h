#pragma once

/**
 * @file torch_api.h
 * @brief Minimal stand-in for <torch/extension.h>.
 *
 * torch/extension.h drags in the whole C++ frontend (autograd, nn, pybind11).
 * Under nvcc that header alone costs ~30s per translation unit, and none of it
 * is used here: the ops are registered through TORCH_LIBRARY and only ever
 * touch ATen tensors. Including just the pieces we need drops the fixed header
 * cost to ~11s per translation unit.
 *
 * When reaching for a new tensor factory or op, add its per-operator header
 * (ATen/ops/<op>.h) below rather than pulling in the ATen/ATen.h umbrella,
 * which costs ~5s more on its own.
 */

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
