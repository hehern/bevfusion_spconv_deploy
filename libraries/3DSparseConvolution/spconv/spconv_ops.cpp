#include <limits>
#include <iostream>

#include "spconv_ops.h"
#include "common/check.hpp"
namespace spconv {

/*
  in:
  indices:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  outSpatialShape:vector<int>, size()==3, 输出体素栅格shape,eg:{720, 720, 21}
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
  auto NDim = kernelSize.size();//3
  bool useHash = false;
  auto numAct = indices.shape[0];
  auto coorDim = indices.shape[1] - 1; // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }//27
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }//720*720*21=10886400
  std::string msg = "due to limits of cuda hash, the volume of dense space "
                    "include batch size ";
  msg += "must less than std::numeric_limits<int>::max() = 2e9";
  TV_ASSERT_RT_ERR(outputVolume < std::numeric_limits<int>::max(), msg);
  nv::Tensor indicePairs = nv::Tensor::create(std::vector<int32_t>{2, kernelVolume, numAct}, nv::DataType::Int32);//shape:{2,27,n}
  indicePairs.memset(-1, stream);
  nv::Tensor indiceNum = nv::Tensor::create(std::vector<int32_t>{kernelVolume}, nv::DataType::Int32);//shape:{27}
  indiceNum.memset(0, stream);
  nv::Tensor gridOut = nv::Tensor::create(std::vector<int32_t>{outputVolume}, nv::DataType::Int32);//输出tensor，展平为1维的
  gridOut.memset(-1, stream);
  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{NDim}, nv::DataType::Int32);//output_shape
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(),outSpatialShape.size()*sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream));

  // 参考资料：https://zhuanlan.zhihu.com/p/383299678
  int64_t numActOut = -1;//如果subM类型的spconv，输出actnum和输入actnum是一致的，如果subM为false，则需要计算
  if (subM) {//子流行卷积
    numActOut = create_submconv_indice_pair_cuda(indices, gridOut, indicePairs, indiceNum, ou, outputVolume, stream);
    return {indices, indicePairs, indiceNum};
  } else {//非子流行卷积
    nv::Tensor indicePairUnique = nv::Tensor::create(std::vector<int32_t>{indicePairs.numel / 2 + 1}, nv::DataType::Int32);//N*2*27/2+1
    indicePairUnique.memset(std::numeric_limits<int>::max(), stream);
    nv::Tensor outInds = nv::Tensor::create(std::vector<int32_t>{numAct * kernelVolume, coorDim + 1}, nv::DataType::Int32);//{n*27, 4}
    outInds.memset(0, stream);

    numActOut = create_conv_indice_pair_p1_cuda(indices, indicePairs, indiceNum, indicePairUnique, kernelSize, stride, padding, dilation, outSpatialShape, outputVolume, stream);
    if (numActOut > 0) {
      // auto res = torch::_unique(indicePairUnique);//
      // indicePairUnique = std::get<0>(res);//
      numActOut = create_conv_indice_pair_p2_cuda(indices, outInds, gridOut, indicePairs, indiceNum, indicePairUnique, outSpatialShape, stream);
    }
    // return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
    return {outInds, indicePairs, indiceNum};
  }
}

nv::Tensor indiceConv(nv::Tensor features,    // 输入特征(N,5)
                      nv::Tensor filters,     // eg:权重[16,3,3,3,5],5为输入channel个数，16为输出channel个数
                      nv::Tensor indicePairs, // [2, 27, N]
                      nv::Tensor indiceNum,   // [27]，用于保存卷积核每一个位置上的总的计算的次数
                      bool subM, void* stream) {            // 子流线卷积默认 true
  
  auto numActOut = features.size(0);     // N
  auto kernelVolume = indiceNum.size(0); // 27
  auto numInPlanes = features.size(1);   // 5
  auto numOutPlanes = filters.size(0);   // 16
  auto indicePairNumCpu = indiceNum.to_host();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  nv::Tensor output = nv::Tensor::create(std::vector<int32_t>{numActOut, numOutPlanes}, features.dtype(), features.device());
  output.memset(0, stream);
  filters = filters.view({-1, numInPlanes, numOutPlanes});//view这个操作好像只改变shape的顺序并不改变内存的保存顺序

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2; // 13
  int indicePairMaxSize = numActOut;          // N
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    // torch::mm_out(output, features, filters[indicePairMaxOffset]);//矩阵乘法，此处需要替换为自己的函数
    unsigned short* features_ptr = features.ptr<unsigned short>();//其实是fp16
    unsigned short* weight_ptr = filters.ptr<unsigned short>();//这里需要加个偏移量到filters[indicePairMaxOffset]
    unsigned short* output_ptr = output.ptr<unsigned short>();
    cuda_2d_launch(matrix_multiply_cuda_naive, _stream, numActOut, numOutPlanes, numInPlanes, features_ptr, weight_ptr, output_ptr);
    

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  nv::Tensor inputBuffer = nv::Tensor::create(std::vector<int32_t>{indicePairMaxSize, numInPlanes}, features.dtype(), features.device());
  inputBuffer.memset(0, stream);
  nv::Tensor outputBuffer = nv::Tensor::create(std::vector<int32_t>{indicePairMaxSize, numOutPlanes}, features.dtype(), features.device());
  outputBuffer.memset(0, stream);

  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;

  // 按照rulebook逐卷积核元素计算
  for (int i = 0; i < kernelVolume; ++i) {//27
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];//表示第i个卷积核元素对应激活输入输出对个数(count)
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }

    sparse_gather_cuda(inputBuffer, features, indicePairs[0][i], nHot);//根据indicePairs中的vin查找到对应的输入voxels的值，并保存在inputBuffer
    torch::mm_out(outputBuffer, inputBuffer, filters[i]);//矩阵乘法
    sparse_scatter_add_cuda(outputBuffer, output, indicePairs[1][i], nHot);//将结果填充到output中去

  }

  return output;
}

} // namespace spconv
