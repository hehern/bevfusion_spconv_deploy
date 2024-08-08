#include <limits>
#include <iostream>

#include "spconv_ops.h"
#include "common/check.hpp"
namespace spconv {

/*
  indices:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  outSpatialShape:vector<int>, size()==3, 输出体素栅格shape,eg:{720, 720, 21}
  spatialShape:vector<int>, size()==3, 输入体素栅格shape,eg:{1440, 1440, 41}
  kernelSize:vector<int>, size()==3,eg:{3, 3, 3}
  stride:vector<int>, size()==3,eg:{1, 1, 1}
  padding:vector<int>, size()==3,eg:{1, 1, 1}
  dilation:vector<int>, size()==3,eg:{1, 1, 1}
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

// nv::Tensor indiceConv(nv::Tensor features,    // 输入特征(N,5)
//                       nv::Tensor filters,     // 权重(27*16*32),16为输入channel个数，32为输出channel个数
//                       nv::Tensor indicePairs, // [2, 27, N]
//                       nv::Tensor indiceNum,   // [27]，用于保存卷积核每一个位置上的总的计算的次数
//                       bool subM) {            // 子流线卷积默认 true
  
//   auto numActOut = features.size(0);     //N
//   auto kernelVolume = indiceNum.size(0); //27
//   auto ndim = filters.dim() - 2;         //
//   auto numInPlanes = features.size(1);
//   auto numOutPlanes = filters.size(ndim + 1);
//   auto indicePairNumCpu = indiceNum.to({torch::kCPU});

//   nv::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
//   filters = filters.view({-1, numInPlanes, numOutPlanes});

//   // init for subM
//   int indicePairMaxOffset = kernelVolume / 2;
//   int indicePairMaxSize = numActOut;
//   if (subM) { // the center index of subm conv don't need gather and scatter
//     // add.
//     torch::mm_out(output, features, filters[indicePairMaxOffset]);

//     // get indice pair second max size based on subM symmetric property
//     indicePairMaxSize =
//       *std::max_element(indicePairNumCpu.data_ptr<int>(),
//                         indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
//     if (indicePairMaxSize == 0) {
//       return output;
//     }
//   } else {
//     indicePairMaxSize =
//       *std::max_element(indicePairNumCpu.data_ptr<int>(),
//                         indicePairNumCpu.data_ptr<int>() + kernelVolume);
//   }

//   nv::Tensor inputBuffer =
//       torch::empty({indicePairMaxSize, numInPlanes}, options);
//   nv::Tensor outputBuffer =
//       torch::empty({indicePairMaxSize, numOutPlanes}, options);

//   double totalGatherTime = 0;
//   double totalGEMMTime = 0;
//   double totalSAddTime = 0;
//   // tv::ssprint("first subm gemm time", timer.report() / 1000.0);

//   for (int i = 0; i < kernelVolume; ++i) {
//     auto nHot = indicePairNumCpu.data_ptr<int>()[i];
//     if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
//       continue;
//     }
//     // TODO torch::from_blob is a little slow
//     auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
//                                              {nHot, numOutPlanes}, options);
//     auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr(), {nHot, numInPlanes}, options);


//     sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
//     /* slower than SparseGatherFunctor, may due to int->long conversion
//     auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
//     auto indicePairBlob = torch::from_blob(indicePairLong.data<long>(),
//     {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
//     features, 0, indicePairBlob);*/
   
//     torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
//     sparse_scatter_add_cuda(outputBuffer, output, indicePairs[!inverse][i], nHot);

//   }

//   return output;
// }

} // namespace spconv
