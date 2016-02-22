# im2col

a = [1 4
     2 5
     3 6]
b = [7 10
     8 11
     9 12]
x = [a b]
cols = im2col(x,size(a),"distinct")
@test cols == [vec(a) vec(b)]
@test cols == im2col!(similar(cols),x,size(a),size(a))
@test_throws DimensionMismatch im2col!(Array(Int,1,2),x,size(a),size(a))

x = randn(4,1,7)
sz = (2,1,3)
f = rand(sz)
@test_approx_eq  vec(f)' * im2col(x,sz,"sliding")  imfilter(x,f,"inner")

x = reshape(1:(3*7),3,7)
@test x[:] == im2col(x,(1,1))[:]
@test im2col(x,(3,1),(1,3)) == x[:,1:3:end]
@test_throws DimensionMismatch im2col(x,(3,1),(1,4))
@test_throws DimensionMismatch im2col(x,(4,1))
@test_throws ArgumentError im2col(x,(1,1),"foo")