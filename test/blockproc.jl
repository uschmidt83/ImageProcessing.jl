## im2col

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



## col2im

szI = (4,3,2)
szW  = (2,1,2)
N = length(szI)
onetuple = ntuple(d->1,N)
I = randn(szI)

# identity
@test I == col2im(reshape(I,1,length(I)),onetuple,szI,"distinct")
@test I == col2im(reshape(I,1,length(I)),onetuple,szI,"sliding")
@test I == col2im(reshape(I,1,length(I)),onetuple,szI,onetuple)

# distinct
cols = im2col(I,szW,"distinct")
@test I == col2im(cols,szW,szI,"distinct")
@test I == col2im(cols,szW,szI,szW)
@test I == col2im!(zeros(I),cols,szW,szW)

# sliding with averaging
cols = im2col(I,szW,"sliding")
@test I == col2im(cols,szW,szI)
@test I == col2im(cols,szW,szI,"sliding")
@test I == col2im(cols,szW,szI,"sliding","average")
@test I == col2im(cols,szW,szI,onetuple)
@test I == col2im(cols,szW,szI,ntuple(d->1,N),+,0,/)

# sliding
I = rand(3,3)
cols = im2col(I,(2,2),"sliding")
w = [1 2 1; 2 4 2; 1 2 1]
ref = I .* w
@test_approx_eq  ref  col2im(cols,(2,2),(3,3),"sliding","sum")
@test_approx_eq  ref  col2im(cols,(2,2),(3,3),(1,1),+,0,nothing)
@test_approx_eq  ref  col2im!(zeros(I),cols,(2,2),(1,1),+)
ref = I .^ w
@test_approx_eq  ref  col2im(cols,(2,2),(3,3),(1,1),*,1,nothing)
@test_approx_eq  ref  col2im!(ones(I),cols,(2,2),(1,1),*)
ref = I
@test_approx_eq  ref  col2im(cols,(2,2),(3,3),"sliding","average")
@test_approx_eq  ref  col2im(cols,(2,2),(3,3),(1,1),+,0,/)
ref = I .* w - w
@test_approx_eq  ref  col2im(cols,(2,2),(3,3),(1,1),+,0,-)

# errors
I = rand(Int,4,4); cols = im2col(I,(2,2),"sliding")
@test_throws ArgumentError col2im(cols,(2,2),(4,4),"sliding","average")
I = rand(4,4); cols = im2col(I,(2,2),"sliding")
@test_throws DimensionMismatch col2im(cols,(2,2),(3,4))
@test_throws DimensionMismatch col2im(cols,(2,2),(4,4),"distinct")
@test_throws ArgumentError col2im(cols,(2,2),(4,4),"sliding","foo")
@test_throws ArgumentError col2im(cols,(2,2),(4,4),"foo")
