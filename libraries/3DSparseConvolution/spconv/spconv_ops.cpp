#include "spconv_ops.h"
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
  nv::Tensor indicePairs = nv::Tensor::create({2, kernelVolume, numAct}, nv::DataType::Int32);//shape:{2,27,n}
  indicePairs.memset(-1);
  nv::Tensor indiceNum = nv::Tensor::create({kernelVolume}, nv::DataType::Int32);//shape:{27}
  indiceNum.memset(0);
  nv::Tensor gridOut = nv::Tensor::create({outputVolume}, nv::DataType::Int32);//输出tensor，展平为1维的
  gridOut.memset(-1);
  nv::Tensor ou = nv::Tensor::create({NDim}, nv::DataType::Int32);//输出tensor，展平为1维的
  int ou_host[3] = {outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]};
  ou.copy_from_host(ou_host);

  // 参考资料：https://zhuanlan.zhihu.com/p/383299678
  int64_t numActOut = -1;//如果subM类型的spconv，输出actnum和输入actnum是一致的，如果subM为false，则需要计算
  if (subM) {
    numActOut = create_submconv_indice_pair_cuda(indices, gridOut, indicePairs, indiceNum, ou, false, useHash, stream);
    return {indices, indicePairs, indiceNum};
  } else {
    // auto indicePairUnique = torch::full({indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(), torch::dtype(torch::kInt32).device(indices.device()));
    // nv::Tensor outInds = torch::zeros({numAct * kernelVolume, coorDim + 1}, torch::dtype(torch::kInt32).device(indices.device()));

    // if (indices.data->device) {
    //   numActOut = create_conv_indice_pair_p1_cuda(indices, indicePairs, indiceNum, indicePairUnique, kernelSize, stride, padding, dilation, outSpatialShape);
    //   if (numActOut > 0) {
    //     auto res = torch::_unique(indicePairUnique);
    //     indicePairUnique = std::get<0>(res);
    //     numActOut = create_conv_indice_pair_p2_cuda(indices, outInds, gridOut, indicePairs, indiceNum, indicePairUnique, outSpatialShape, false, useHash);
    //   }
    // } else {
    //   TV_THROW_INVALID_ARG("not cuda type");
    // }
    // return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}

// nv::Tensor indiceConv(nv::Tensor features, nv::Tensor filters,
//                       nv::Tensor indicePairs, nv::Tensor indiceNum,
//                       int64_t numActOut, int64_t _subM) {
//   auto kernelVolume = indiceNum.size(0);

//   // auto timer = spconv::CudaContextTimer<>();

//   bool subM = _subM != 0;
//   auto device = features.device().type();
//   auto ndim = filters.dim() - 2;
//   auto numInPlanes = features.size(1);
//   auto numOutPlanes = filters.size(ndim + 1);
//   auto indicePairNumCpu = indiceNum.to({torch::kCPU});

//   auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
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
//     auto inputBufferBlob =
//         torch::from_blob(inputBuffer.data_ptr(), {nHot, numInPlanes}, options);

//     if (device == torch::kCUDA) {
//       sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
//       /* slower than SparseGatherFunctor, may due to int->long conversion
//       auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
//       auto indicePairBlob = torch::from_blob(indicePairLong.data<long>(),
//       {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
//       features, 0, indicePairBlob);*/
//     } else {
//       TV_THROW_INVALID_ARG("not cuda type");
//     }
//     // totalGatherTime += timer.report() / 1000.0;
//     torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
//     // totalGEMMTime += timer.report() / 1000.0;

//     if (device == torch::kCUDA) {
//       sparse_scatter_add_cuda(outputBuffer, output, indicePairs[!inverse][i], nHot);
//     } else {
//       TV_THROW_INVALID_ARG("not cuda type");
//     }
//     // totalSAddTime += timer.report() / 1000.0;
//   }
//   // tv::ssprint(totalGatherTime, totalGEMMTime, totalSAddTime);
//   return output;
// }

} // namespace spconv
