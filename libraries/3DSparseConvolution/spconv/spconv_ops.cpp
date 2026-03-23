#include <limits>
#include <iostream>
#include <algorithm>

#include <cuda_fp16.h>

#include "spconv_ops.h"
#include "common/check.hpp"
#include "common/timer.hpp"
namespace spconv {

/*
  in:
  indices:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  outSpatialShape:vector<int>, size()==3, 输出体素栅格shape,eg:{720, 720, 21},xyz的顺序
  spatialShape:vector<int>, size()==3, 输入体素栅格shape,eg:{1440, 1440, 41}
  kernelSize:vector<int>, size()==3,eg:{3, 3, 3}
  stride:vector<int>, size()==3,eg:{1, 1, 1}
  padding:vector<int>, size()==3,eg:{1, 1, 1}
  dilation:vector<int>, size()==3,eg:{1, 1, 1}
  out:
  indices: nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  indicePairs: shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即grid的一维index
  indiceNum: nv::Tensor, shape:{27},对应的是rule_book中的count
*/
std::vector<nv::Tensor>
getIndicePairs(nv::Tensor indices,
               std::vector<int> outSpatialShape,
               std::vector<int> spatialShape,
               std::vector<int> kernelSize, std::vector<int> stride,
               std::vector<int> padding, std::vector<int> dilation,
               bool subM, void* stream) {
  int64_t NDim = kernelSize.size();//3
  auto numAct = indices.shape[0];//type:long int
  auto coorDim = indices.shape[1] - 1; // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(int64_t(kernelSize.size()) == coorDim, "error");
  TV_ASSERT_RT_ERR(int64_t(outSpatialShape.size()) == coorDim, "error");
  TV_ASSERT_RT_ERR(int64_t(stride.size()) == coorDim, "error");
  TV_ASSERT_RT_ERR(int64_t(padding.size()) == coorDim, "error");
  TV_ASSERT_RT_ERR(int64_t(dilation.size()) == coorDim, "error");
  int64_t kernelVolume = kernelSize[0];
  for (size_t i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }//27
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (size_t i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }//720*720*21=10886400
  std::string msg = "due to limits of cuda hash, the volume of dense space "
                    "include batch size ";
  msg += "must less than std::numeric_limits<int>::max() = 2e9";
  TV_ASSERT_RT_ERR(outputVolume < std::numeric_limits<int64_t>::max(), msg);
  nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{2, kernelVolume, numAct}, nv::DataType::Int32);//shape:{2,27,n}
  indicePairs.fill<int32_t>(-1);
  nv::Tensor indiceNum = nv::Tensor::create(std::vector<int64_t>{kernelVolume}, nv::DataType::Int32);//shape:{27}
  indiceNum.fill<int32_t>(0);
  nv::Tensor gridOut = nv::Tensor::create(std::vector<int64_t>{outputVolume}, nv::DataType::Int32);//输出tensor，展平为1维的
  gridOut.fill<int32_t>(-1);
  nv::Tensor ou = nv::Tensor::create(std::vector<int64_t>{NDim}, nv::DataType::Int32);//output_shape
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), outSpatialShape.size()*sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream));

  // 参考资料：https://zhuanlan.zhihu.com/p/383299678
  int64_t numActOut = -1;//如果subM类型的spconv，输出actnum和输入actnum是一致的，如果subM为false，则需要计算
  nv::EventTimer timer_;
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  if (subM) {//子流行卷积
    // timer_.start(_stream);
    numActOut = create_submconv_indice_pair_cuda(indices, gridOut, indicePairs, indiceNum, ou, outputVolume, stream);
    // timer_.stop("create_submconv_indice_pair_cuda");
    return {indices, indicePairs, indiceNum};
  } else {//非子流行卷积
    // std::cout << "not subm" << std::endl;
    // checkRuntime(cudaStreamSynchronize(_stream));
    nv::Tensor indicePairUnique = nv::Tensor::create(std::vector<int64_t>{int64_t(indicePairs.numel / 2) + 1}, nv::DataType::Int32);//N*2*27/2+1
    indicePairUnique.fill<int32_t>(std::numeric_limits<int32_t>::max());
    nv::Tensor outInds = nv::Tensor::create(std::vector<int64_t>{numAct * kernelVolume, coorDim + 1}, nv::DataType::Int32);//{n*27, 4}，这里定义的数据量太大了，后面n*26根本没用
    outInds.fill<int32_t>(0);
    // checkRuntime(cudaStreamSynchronize(_stream));
    // timer_.start(_stream);
    numActOut = create_conv_indice_pair_p1_cuda(indices, indicePairs, indiceNum, indicePairUnique, kernelSize, stride, padding, dilation, outSpatialShape, outputVolume, stream);
    // timer_.stop("create_conv_indice_pair_p1_cuda");
    // std::cout << "not subm, rulebook 1, numActOut = " << numActOut << std::endl;
    if (numActOut > 0) {
      // timer_.start(_stream);
      nv::Tensor indicePairUnique_new = find_unique_elements_cuda(indicePairUnique, stream);//挑出tensor中的独立不重复元素,并按照升序排列，indicePairUnique中保存的是vout即输出voxel grid的一维index
      // timer_.stop("find_unique_elements_cuda");
      // std::cout << "not subm, rulebook 2, find_unique_elements_cuda done" << std::endl;
      // timer_.start(_stream);
      numActOut = create_conv_indice_pair_p2_cuda(indices, outInds, gridOut, indicePairs, indiceNum, indicePairUnique_new, outSpatialShape, stream);
      // timer_.stop("create_conv_indice_pair_p2_cuda");
      // std::cout << "not subm, rulebook 2, numActOut = " << numActOut << std::endl;
    }
    nv::Tensor finalOutInds = nv::Tensor::from_data(outInds.ptr<int>(), std::vector<int64_t>{numActOut, coorDim + 1}, nv::DataType::Int32);//切片，这地方用from_data有点浪费了，可以优化
    // return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};//at::Tensor slice(int64_t dim=0, ::std::optional<int64_t> start=::std::nullopt, ::std::optional<int64_t> end=::std::nullopt, int64_t step=1) const;
    return {finalOutInds, indicePairs, indiceNum};
  }
}

nv::Tensor indiceConv(nv::Tensor features,    // 输入特征(N,5)
                      nv::Tensor filters,     // eg:权重[3*3*3,5,16],5为输入channel个数，16为输出channel个数
                      nv::Tensor indicePairs, // [2, 27, N]
                      nv::Tensor indiceNum,   // [27]，用于保存卷积核每一个位置上的总的计算的次数
                      int64_t numActOut,
                      bool subM, void* stream) {            // 子流线卷积默认 true
  
  auto kernelVolume = indiceNum.size(0); // 27
  auto numInPlanes = features.size(1);   // 5
  auto numOutPlanes = filters.size(2);   // 16
  auto numActIn = indicePairs.size(2);

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  nv::EventTimer timer_;

  auto indicePairNumCpu = indiceNum.to_host();

  nv::Tensor output = nv::Tensor::create(std::vector<int64_t>{numActOut, numOutPlanes}, features.dtype(), features.device());
  output.memset(0, stream);

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2; // 13
  int indicePairMaxSize = numActOut;          // N
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    // timer_.start(_stream);
    matrix_multiply_cuda(features, filters, output, numActOut, numOutPlanes, numInPlanes, indicePairMaxOffset, stream);
    // timer_.stop("matrix_multiply_cuda");

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.ptr<int>(),
                        indicePairNumCpu.ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.ptr<int>(),
                        indicePairNumCpu.ptr<int>() + kernelVolume);
  }

  nv::Tensor inputBuffer = nv::Tensor::create(std::vector<int64_t>{indicePairMaxSize, numInPlanes}, features.dtype(), features.device());
  inputBuffer.memset(0, stream);
  nv::Tensor outputBuffer = nv::Tensor::create(std::vector<int64_t>{indicePairMaxSize, numOutPlanes}, features.dtype(), features.device());
  outputBuffer.memset(0, stream);


  // 按照rulebook逐卷积核元素计算
  for (int i = 0; i < kernelVolume; ++i) {//27
    auto nHot = indicePairNumCpu.ptr<int>()[i];//表示第i个卷积核元素对应激活输入输出对个数(count)
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }

    // timer_.start(_stream);
    sparse_gather_cuda(inputBuffer, features, indicePairs, nHot, i*numActIn, stream);//根据indicePairs中的vin查找到对应的输入voxels的值，并保存在inputBuffer
    // timer_.stop("sparse_gather_cuda");
    // timer_.start(_stream);
    matrix_multiply_cuda(inputBuffer, filters, outputBuffer, nHot, numOutPlanes, numInPlanes, i, stream);//gemm
    // timer_.stop("matrix_multiply_cuda");
    // timer_.start(_stream);
    sparse_scatter_add_cuda(outputBuffer, output, indicePairs, nHot, (kernelVolume+i)*numActIn, stream);//将结果填充到output中去
    // timer_.stop("sparse_scatter_add_cuda");
  }

  // auto outputHost = output.to_host(stream);
  // std::cout << "indiceConv output tohost" << std::endl;
  return output;
}

void printFeatures(nv::Tensor features, void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  checkRuntime(cudaStreamSynchronize(_stream));

  auto featuresHost = features.to_host(stream);
  auto f_h_ptr = featuresHost.ptr<half>();
  printf("features numel = %d\n", featuresHost.numel);
  for(size_t i=0; i<featuresHost.numel; i++) {
    float f_f = __half2float(f_h_ptr[i]);
    printf("%f,", f_f);
    if ((i+1)%features.shape[1] == 0) {
      printf("\n");
    }

  }
  printf("\n");
}

} // namespace spconv
