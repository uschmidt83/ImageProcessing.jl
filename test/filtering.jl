# imfiltermtx

x = randn(7,4,5)
k = randn(5,1,4)

for border in ("replicate", "circular", "reflect", "symmetric", "value", "inner")
  @test_approx_eq  imfilter(x,k,border)  imfiltermtx(size(x),k,border)*vec(x)
end
@test_approx_eq  imfilter(x,k)  imfiltermtx(size(x),k)*vec(x)
@test_throws ArgumentError imfiltermtx(size(x),k,"value",1)
