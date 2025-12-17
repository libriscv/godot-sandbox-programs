module {
  func.func @test_assign() -> tensor<f32> {
    %c0 = stablehlo.constant dense<5.0> : tensor<f32>
    %c1 = stablehlo.constant dense<10.0> : tensor<f32>
    %v2 = stablehlo.add %c0, %c1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    stablehlo.return %v2 : tensor<f32>
  }
}
