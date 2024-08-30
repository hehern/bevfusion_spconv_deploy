// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef REORDERING_CU_H_
#define REORDERING_CU_H_
// #include <THC/THCAtomics.cuh>
// #include <THC/THCNumerics.cuh>
#include <cuda_fp16.h>
// #include <tensorview/kernel_utils.h>

#if PYTORCH_VERSION < 10500
#define TH_ATOMIC_ADD atomicAdd
#else
#define TH_ATOMIC_ADD gpuAtomicAdd
#endif

// see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
namespace spconv {

__global__ void gatherGenericKernel(half *buffer, const half *features,
                                    const int32_t *indices, int size,
                                    int numPlanes) {
  int ILPStrideX[NumILP];
  int32_t inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              features[inds[ilp] + iy];
      }
    }
  }
}

__global__ void gatherVecKernel(half *buffer, const half *features,
                                const int32_t *indices, int size, int numPlanes) {
  int ILPStrideX[NumILP];
  int32_t inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          reinterpret_cast<VecType *>(
              buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              reinterpret_cast<const VecType *>(features)[inds[ilp] + iy];
      }
    }
  }
}

__global__ void gatherVecBlockKernel(half *buffer, const half *features,
                                     const int32_t *indices, int size,
                                     int numPlanes) {
  int ILPStrideX[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  features += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      reinterpret_cast<VecType *>(
          buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y] =
          reinterpret_cast<const VecType *>(
              features)[indices[ix + ILPStrideX[ilp]] * numPlanes +
                        threadIdx.y];
    }
  }
}

__global__ void scatterAddGenericKernel(half *outFeatures, const half *buffer,
                                        const int32_t *indices, int size,
                                        int numPlanes) {
  int ILPStrideX[NumILP];
  int32_t inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size) {
          outFeatures[inds[ilp] + iy] +=
              buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy];
        }
      }
    }
  }
}

__global__ void scatterAddVecBlockKernel(half *outFeatures, const half *buffer,
                                         const int32_t *indices, int size,
                                         int numPlanes) {
  int ILPStrideX[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(half);
  constexpr int vecloadHalf2Factor = sizeof(VecType) / sizeof(__half2);

#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  outFeatures += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;
  half buf[vecloadFactor];
  half buf2[vecloadFactor];
  int32_t idx;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idx = indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y;
      reinterpret_cast<VecType *>(buf)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idx];
      reinterpret_cast<VecType *>(buf2)[0] = reinterpret_cast<const VecType *>(
          buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y];
      if (std::is_same<half, at::Half>::value) {
#if __CUDA_ARCH__ >= 530
#pragma unroll
        for (int i = 0; i < vecloadHalf2Factor; i++) {
          reinterpret_cast<__half2 *>(buf)[i] =
              __hadd2(reinterpret_cast<__half2 *>(buf)[i],
                      reinterpret_cast<__half2 *>(buf2)[i]);
        }
#else
#pragma unroll
        for (int i = 0; i < vecloadFactor; i++) {
          buf[i] += buf2[i];
        }
#endif
      } else {
#pragma unroll
        for (int i = 0; i < vecloadFactor; i++) {
          buf[i] += buf2[i];
        }
      }
      reinterpret_cast<VecType *>(outFeatures)[idx] =
          reinterpret_cast<VecType *>(buf)[0];
    }
  }
}

__global__ void scatterAddBlockKernel(half *outFeatures, const half *buffer,
                                      const int32_t *indices, int size,
                                      int numPlanes) {
  int ILPStrideX[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  outFeatures += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      outFeatures[indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y] +=
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y];
    }
  }
}


} // namespace spconv

#undef TH_ATOMIC_ADD

#endif