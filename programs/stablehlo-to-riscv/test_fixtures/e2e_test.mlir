module {
  func.func @e2e_test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %v0 = stablehlo.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %c0 = stablehlo.constant dense<100.0> : tensor<f32>
    %c1 = stablehlo.constant dense<0.0> : tensor<f32>
    %cmp = stablehlo.compare GT, %v0, %c1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %v1 = stablehlo.select %cmp, %c0, %v0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    stablehlo.return %v1 : tensor<f32>
  }
}
