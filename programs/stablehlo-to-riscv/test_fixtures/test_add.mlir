module {
  func.func @test_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %v0 = stablehlo.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    stablehlo.return %v0 : tensor<f32>
  }
}
