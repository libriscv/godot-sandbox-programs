module {
  func.func @e2e_test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %v0 = stablehlo.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %c0 = stablehlo.constant dense<100> : tensor<f32>
    %v1 = stablehlo.if %v0 -> (tensor<f32>) { 
      stablehlo.return %c0 : tensor<f32>
    } : (tensor<f32>) { 
      stablehlo.return %v0 : tensor<f32>
    }
    stablehlo.return %v1 : tensor<f32>
  }
}
