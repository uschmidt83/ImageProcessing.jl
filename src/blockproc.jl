"""
```
cols = im2col(img, blocksize, [blocktype or strides])
```

creates a copy of all image blocks of size `blocksize` and arranges them as
the columns of output array `cols`. The third parameter can either be a tuple
of `strides` or a `blocktype` string (`"sliding"`, `"distinct"`); the default
value is `"sliding"`, which corresponds to a stride of 1 in all dimensions.

This function is similar to Matlab's `im2col`, but supports N-D images.
See http://www.mathworks.com/help/images/ref/im2col.html
"""

function im2col{T,N}(A::AbstractArray{T,N}, sz::NTuple{N,Int}, blocktype::AbstractString="sliding")    
  if blocktype == "sliding"
    return im2col(A,sz,tuple(ones(Int,N)...))
  elseif blocktype == "distinct"
    return im2col(A,sz,sz)
  else
    throw(ArgumentError("Invalid 'blocktype' (must be 'sliding' or 'distinct')."))
  end
end

function im2col{T,N}(A::AbstractArray{T,N}, sz::NTuple{N,Int}, stride::NTuple{N,Int})
  check_im2col(size(A),sz,stride)
  ncols::Int = mapreduce( d -> 1 + div(size(A,d)-sz[d], stride[d]), *, 1:N)
  im2col!(Array(T,prod(sz),ncols), A, sz, stride)
end

"""
```
im2col!(cols, img, blocksize, [strides])
```

stores all image blocks of size `blocksize` in the pre-allocated output `cols`. 

See also: `im2col`.
"""

@generated function im2col!{T,N}(cols::AbstractArray{T,2}, A::AbstractArray{T,N}, sz::NTuple{N,Int}, stride::NTuple{N,Int}=tuple(ones(Int,N)...))
  quote
    check_im2col(size(A),sz,stride,size(cols))
    c = 0
    @inbounds begin
      @nloops $N i d->1:stride[d]:(size(A,d)-sz[d]+1) begin
        r = 0; c += 1
        @nloops $N j d->0:(sz[d]-1) begin
          r += 1
          cols[r,c] = @nref $N A d->i_d+j_d
        end
      end
    end
    cols
  end
end

function check_im2col{N}(szA::NTuple{N,Int},szB::NTuple{N,Int},stride::NTuple{N,Int},szcols::Union{Void,NTuple{2,Int}}=nothing)
  all(d -> mod(szA[d]-szB[d],stride[d]) == 0, 1:N) || throw(DimensionMismatch("Image and window sizes not compatible with strides."))
  all(d -> szA[d] >= szB[d], 1:N) || throw(DimensionMismatch("Window size larger than image."))
  if szcols != nothing
    ncols = mapreduce( d -> 1 + div(szA[d]-szB[d], stride[d]), *, 1:N)
    szcols == (prod(szB),ncols) || throw(DimensionMismatch("Incorrect size of output array 'cols'."))
  end
end
