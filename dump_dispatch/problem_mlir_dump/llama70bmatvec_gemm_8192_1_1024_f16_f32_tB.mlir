#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<8192x1024xf16>, %arg1: tensor<1x1024xf16>) -> tensor<8192x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8192x1xf32>) -> tensor<8192x1xf32>
    %2 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8192x1024xf16>, tensor<1x1024xf16>) outs(%1 : tensor<8192x1xf32>) -> tensor<8192x1xf32>
    return %2 : tensor<8192x1xf32>
  }
}
