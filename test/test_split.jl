using Test
using AutoComputationalGraphTuning

@testset "leading_colons" begin
    # 1D array: should return an empty tuple
    A = AutoComputationalGraphTuning
    x1 = rand(5)
    @test A.leading_colons(x1) == ()
    @test x1[A.leading_colons(x1)..., 3] == x1[3]

    # 2D array: should return (:) for slicing rows or columns
    x2 = rand(4, 6)
    @test A.leading_colons(x2) == (:,)
    @test x2[A.leading_colons(x2)..., 2] == x2[:, 2]

    # 3D array: should return (:, :) for slicing first two dims
    x3 = rand(2, 3, 4)
    @test A.leading_colons(x3) == (:, :)
    @test x3[A.leading_colons(x3)..., 1] == x3[:, :, 1]
end