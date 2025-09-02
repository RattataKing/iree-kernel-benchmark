module {
  func.func @main(%arg0: tensor<4096x2560xf16>, %arg1: tensor<2560x640xf16>) -> tensor<4096x640xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x640xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x2560xf16>, tensor<2560x640xf16>) outs(%1 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
    return %2 : tensor<4096x640xf32>
  }
}
