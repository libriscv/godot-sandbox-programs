module {
  func.func @test_simple() -> tensor<f32> {
    %c0 = stablehlo.constant dense<42> : tensor<f32>
    stablehlo.return %c0 : tensor<f32>
  }
}
