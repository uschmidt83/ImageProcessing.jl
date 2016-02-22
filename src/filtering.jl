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
  npixels = prod(szimg)
  idximg = reshape(1:npixels,szimg)
  if border in ("replicate", "circular", "reflect", "symmetric", "value")
    if border == "value"
      value = convert(T, value)
      value == zero(T) || throw(ArgumentError("Only value '0' supported for border condition 'value'."))
    end
    prepad  = ntuple(d->div(size(k,d)-1, 2), N)
    postpad = ntuple(d->div(size(k,d),   2), N)
    idximg = padarray(idximg, prepad, postpad, border, value)
  elseif border == "inner" # nothing to do    
  else
    throw(ArgumentError("Border condition '$border' not supported."))
  end

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
