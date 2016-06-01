"""
```
M = imfiltermtx(imgsize, kernel, [border, value])
```

creates a sparse matrix such that matrix-vector multiplication is identical to
calling `Images.imfilter` with the same options:

`imfiltermtx(size(img),kernel,border,value) * vec(img)  ≈  vec(imfilter(img,kernel,border,value))`

See `Images.imfilter` for an explanation of the available options.
Note that boundary condition `"value"` does not support any `value`s other than 0.
"""

function imfiltermtx{T,N}(szimg::NTuple{N,Int}, k::AbstractArray{T,N}, border::AbstractString="replicate", value=zero(T))
  border != "value" || convert(T,value) == zero(T) || throw(ArgumentError("Only value '0' supported for border condition 'value'."))
  npixels = prod(szimg)
  idximg = reshape(1:npixels,szimg)
  idximg = padforfilter(idximg, size(k), border, value)

  cols::Matrix{Int} = im2col(idximg,size(k),"sliding")
  nelems,ncols  = size(cols)
  I::Vector{Int} = vec(repmat(reshape(1:ncols,1,ncols),nelems))
  J::Vector{Int} = vec(cols)
  V::Vector{T}   = repmat(vec(k),ncols)
  if border == "value"
    valid = find(J)
    I,J,V = map(x->x[valid],(I,J,V))
  end

  sparse(I,J,V,ncols,npixels)
end



"""
```
otf = psf2otf(psf, [outsize])
```

creates the optical transfer function (`otf`) from the provided point-spread function (`psf`).
The tuple `outsize` can be used to specify the size of the `otf` array; by default, `outsize = size(psf)`.

The convolution theorem relates convolution with the `psf` in the spatial domain to
multiplication with the `otf` in the frequency domain:

`Images.imfilter(x,Images.reflect(psf),"circular") ≈ real(ifft( fft(x) .* psf2otf(psf,size(x)) ))`

This function is similar to Matlab's `psf2otf`.
See http://www.mathworks.com/help/images/ref/psf2otf.html
"""

function psf2otf{T,N}(psf::AbstractArray{T,N}, outsz::NTuple{N,Int}=size(psf))
  psfsz = size(psf)
  if psfsz != outsz
    pad = map(-,outsz,psfsz)
    all(x -> x >= 0, pad) || throw(DimensionMismatch("psf too large for outsz."))
    psf = padarray(psf,tuple(zeros(Int,N)...),pad,"value",T(0))
  end
  shift = map(x -> -floor(Int,x/2), psfsz)
  fft(circshift(psf,shift))
end



"""
```
imgpad = padforfilter(img, kernelsize, [border, value])
```

returns a padded copy of array `img` such that the following holds true:

`imfilter(padforfilter(img,size(kernel),border,value),kernel,"inner") == imfilter(img,kernel,border,value)`

See `Images.imfilter` for an explanation of options `border` and `value`.
"""

function padforfilter{T,N}(A::AbstractArray{T,N}, sz::NTuple{N,Int}, border::AbstractString="replicate", value=zero(T))
  if border in ("replicate", "circular", "reflect", "symmetric", "value")
    prepad  = ntuple(d->div(sz[d]-1, 2), N)
    postpad = ntuple(d->div(sz[d],   2), N)
    A = padarray(A, prepad, postpad, border, convert(T,value))
  elseif border == "inner" # nothing to do    
  else
    throw(ArgumentError("Border condition '$border' not supported."))
  end
  A
end