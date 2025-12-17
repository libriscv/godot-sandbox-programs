module {
  func.func @test_assign() -> tensor<f32> {
    %c0 = stablehlo.constant dense<5> : tensor<f32>
    %c1 = stablehlo.constant dense<10> : tensor<f32>
    %v0 = stablehlo.copy %c0 : tensor<f32>
    %v1 = stablehlo.copy %c1 : tensor<f32>
    %v2 = stablehlo.add %v0, %v1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    stablehlo.return %v2 : tensor<f32>
  }
}
