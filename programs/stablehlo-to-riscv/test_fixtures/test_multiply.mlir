module {
  func.func @test_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %v0 = stablehlo.multiply %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    stablehlo.return %v0 : tensor<f32>
  }
}
