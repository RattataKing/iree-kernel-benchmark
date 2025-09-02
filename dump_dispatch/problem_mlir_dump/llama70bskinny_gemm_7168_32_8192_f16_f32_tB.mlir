#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<7168x8192xf16>, %arg1: tensor<32x8192xf16>) -> tensor<7168x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<7168x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7168x32xf32>) -> tensor<7168x32xf32>
    %2 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<7168x8192xf16>, tensor<32x8192xf16>) outs(%1 : tensor<7168x32xf32>) -> tensor<7168x32xf32>
    return %2 : tensor<7168x32xf32>
  }
}
