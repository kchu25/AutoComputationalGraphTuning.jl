using Test
using AutoComputationalGraphTuning
using Random: MersenneTwister

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

@testset "train_val_test_split basic functionality" begin
    A = AutoComputationalGraphTuning
    # Simple 2D data, 1D labels
    X = reshape(1:100, 1, 100)
    Y = reshape(1:100, 1, 100)
    data = (X = X, Y = Y)
    splits, splits_indices = A.train_val_test_split(data; train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, rng=MersenneTwister(42))
    @test size(splits.train.X, 2) + size(splits.val.X, 2) + size(splits.test.X, 2) == 100
    @test size(splits.train.X, 2) == 70
    @test size(splits.val.X, 2) == 20
    @test size(splits.test.X, 2) == 10
    # Check that indices match between X and Y
    @test all(splits.train.X[1, :] .== splits.train.Y[1, :])
end

@testset "train_val_test_split error handling" begin
    A = AutoComputationalGraphTuning
    X = rand(5, 10)
    Y = rand(1, 9)  # mismatched
    data = (X = X, Y = Y)
    @test_throws ArgumentError A.train_val_test_split(data)
end
