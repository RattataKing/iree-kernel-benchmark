#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<8192x2560xi8>, %arg1: tensor<640x2560xi8>) -> tensor<8192x640xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<8192x640xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<8192x640xi32>) -> tensor<8192x640xi32>
    %2 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8192x2560xi8>, tensor<640x2560xi8>) outs(%1 : tensor<8192x640xi32>) -> tensor<8192x640xi32>
    return %2 : tensor<8192x640xi32>
  }
}
