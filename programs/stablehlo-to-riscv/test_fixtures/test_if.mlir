module {
  func.func @test_if(%arg0: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<10> : tensor<f32>
    %v0 = stablehlo.add %arg0, %c0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %v1 = stablehlo.if %v0 -> (tensor<f32>) { 
      %c1 = stablehlo.constant dense<100> : tensor<f32>
      stablehlo.return %c1 : tensor<f32>
    } : (tensor<f32>) { 
      %c2 = stablehlo.constant dense<0> : tensor<f32>
      stablehlo.return %c2 : tensor<f32>
    }
    stablehlo.return %v1 : tensor<f32>
  }
}
