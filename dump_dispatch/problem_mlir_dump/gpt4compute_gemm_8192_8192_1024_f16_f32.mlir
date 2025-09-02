module {
  func.func @main(%arg0: tensor<8192x1024xf16>, %arg1: tensor<1024x8192xf16>) -> tensor<8192x8192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8192x1024xf16>, tensor<1024x8192xf16>) outs(%1 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
    return %2 : tensor<8192x8192xf32>
  }
}
