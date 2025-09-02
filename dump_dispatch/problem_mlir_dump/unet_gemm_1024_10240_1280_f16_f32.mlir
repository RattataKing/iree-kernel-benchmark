module {
  func.func @main(%arg0: tensor<1024x1280xf16>, %arg1: tensor<1280x10240xf16>) -> tensor<1024x10240xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x10240xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x10240xf32>) -> tensor<1024x10240xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1280xf16>, tensor<1280x10240xf16>) outs(%1 : tensor<1024x10240xf32>) -> tensor<1024x10240xf32>
    return %2 : tensor<1024x10240xf32>
  }
}
