module {
  func.func @main(%arg0: tensor<2048x65536xf16>, %arg1: tensor<65536x8192xf16>) -> tensor<2048x8192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x8192xf32>) -> tensor<2048x8192xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<2048x65536xf16>, tensor<65536x8192xf16>) outs(%1 : tensor<2048x8192xf32>) -> tensor<2048x8192xf32>
    return %2 : tensor<2048x8192xf32>
  }
}
