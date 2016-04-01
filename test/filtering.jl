x = randn(7,4,5)
k = randn(5,1,4)


# imfiltermtx

for border in ("replicate", "circular", "reflect", "symmetric", "value", "inner")
  @test_approx_eq  imfilter(x,k,border)  imfiltermtx(size(x),k,border)*vec(x)
end
@test_approx_eq  imfilter(x,k)  imfiltermtx(size(x),k)*vec(x)
@test_throws ArgumentError imfiltermtx(size(x),k,"value",1)


# psf2otf

@test_approx_eq  imfilter(x,Images.reflect(k),"circular")  real(ifft( fft(x) .* psf2otf(k,size(x)) ))
@test psf2otf(k) == psf2otf(k,size(k))
@test_throws DimensionMismatch psf2otf(k,(size(k)[1]-1,size(k)[2:end]...))
