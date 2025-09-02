module {
  func.func @main(%arg0: tensor<128x2048xf16>, %arg1: tensor<2048x1280xf16>) -> tensor<128x1280xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x1280xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x1280xf32>) -> tensor<128x1280xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x2048xf16>, tensor<2048x1280xf16>) outs(%1 : tensor<128x1280xf32>) -> tensor<128x1280xf32>
    return %2 : tensor<128x1280xf32>
  }
}
