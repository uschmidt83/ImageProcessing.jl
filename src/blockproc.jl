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
    return im2col(A, sz, tuple(ones(Int,N)...))
  elseif blocktype == "distinct"
    return im2col(A, sz, sz)
  else
    throw(ArgumentError("Invalid 'blocktype' (must be 'sliding' or 'distinct')."))
  end
end

function im2col{T,N}(A::AbstractArray{T,N}, sz::NTuple{N,Int}, stride::NTuple{N,Int})
  check_im2col_col2im(size(A),sz,stride)
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
    check_im2col_col2im(size(A),sz,stride,size(cols))
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


function check_im2col_col2im{N}(szA::NTuple{N,Int}, szB::NTuple{N,Int}, stride::NTuple{N,Int}, szcols::Union{Void,NTuple{2,Int}}=nothing)
  all(d -> stride[d] >= 1, 1:N)                    || throw(DimensionMismatch("All strides must be >= 1."))
  all(d -> szA[d] >= szB[d] >= 1, 1:N)             || throw(DimensionMismatch("Window size too small or large."))
  all(d -> mod(szA[d]-szB[d],stride[d]) == 0, 1:N) || throw(DimensionMismatch("Image and window sizes not compatible with strides."))
  if szcols != nothing
    ncols = mapreduce( d -> 1 + div(szA[d]-szB[d], stride[d]), *, 1:N)
    szcols == (prod(szB),ncols) || throw(DimensionMismatch("Image and window sizes not compatible with 'cols' matrix."))
  end
  nothing
end


"""
```
A = col2im(cols, blocksize, sizeA, [strides, accu_op, accu_init, norm_op] or [blocktype, overlap])
```

uses each column of the matrix `cols` as a block of size `blocksize` to
form the output image `A` of size `sizeA`.

There are two sets of optional parameters:

- `strides`:   tuple of block strides (default is a stride of 1 in all dimensions)
- `accu_op`:   function to accumulate values if blocks overlap (default: +)
- `accu_init`: initial value for all pixels of output `A` (default: 0)
- `norm_op`:   function to normalize each pixel with its count of overlapping blocks (default: /)

or

- `blocktype`: string to denote strides, can be `"sliding"` (default: all strides = 1)
               or `"distinct"` (strides = blocksize)
- `overlap`:   string to denote overlap policy (`accu_op`,`accu_init`,`norm_op`),
               can be `"sum"` (default: +,0,nothing) or `"average"` (+,0,/)

This function is similar to Matlab's `col2im`, but supports N-D images and overlapping blocks.
See http://www.mathworks.com/help/images/ref/col2im.html
"""

function col2im{T,N}(cols::AbstractArray{T,2}, sz::NTuple{N,Int}, szA::NTuple{N,Int}, blocktype::AbstractString="sliding", overlap::AbstractString="average")
  if blocktype == "sliding"
    overlap in ("sum","average") || throw(ArgumentError("Invalid 'overlap' (must be 'sum' or 'average')."))
    return col2im(cols, sz, szA, tuple(ones(Int,N)...), +, zero(T), overlap == "average" ? (/) : nothing)
  elseif blocktype == "distinct"
    return col2im(cols, sz, szA, sz)
  else
    throw(ArgumentError("Invalid 'blocktype' (must be 'sliding' or 'distinct')."))
  end
end

function col2im{T,N}(cols::AbstractArray{T,2}, sz::NTuple{N,Int}, szA::NTuple{N,Int}, stride::NTuple{N,Int}, accu_op::Function=+, accu_init=zero(T), norm_op::Union{Void,Function}=/)
  check_im2col_col2im(szA,sz,stride,size(cols))
  do_normalize = any(d -> stride[d] < sz[d], 1:N) && norm_op != nothing
  !do_normalize || isa(norm_op(cols[1],1),T) ||
    (U = typeof(norm_op(cols[1],1)); throw(ArgumentError("'norm_op' causes type change from $T to $U: try changing the type of 'cols' to $U beforehand.")))
  A = col2im!(fill!(Array{T}(szA),convert(T,accu_init)), cols, sz, stride, accu_op)
  do_normalize && broadcast!(norm_op, A, A, overlap_counts(szA,sz,stride))
  A
end


overlap_counts{N}(szA::NTuple{N,Int}, sz::NTuple{N,Int}, stride::NTuple{N,Int}) =
  col2im( im2col(ones(Int,szA),sz,stride), sz, szA, stride, +, 0, nothing)


"""
```
col2im!(A, cols, blocksize, [strides, accu_op])
```

rearranges the columns of `cols` as blocks of size `blocksize` in the pre-allocated output `A`.

See also: `col2im`.
"""

col2im!{T,N}(A::AbstractArray{T,N}, cols::AbstractArray{T,2}, sz::NTuple{N,Int}, stride::NTuple{N,Int}=tuple(ones(Int,N)...), op::Function=+) =
  col2im!(A,cols,sz,stride,Type{symbol(op)})

@generated function col2im!{T,N,op}(A::AbstractArray{T,N}, cols::AbstractArray{T,2}, sz::NTuple{N,Int}, stride::NTuple{N,Int}, ::Type{Type{op}})
  quote
    check_im2col_col2im(size(A),sz,stride,size(cols))
    c = 0
    @inbounds begin
      @nloops $N i d->1:stride[d]:(size(A,d)-sz[d]+1) begin
        r = 0; c += 1
        @nloops $N j d->0:(sz[d]-1) begin
          r += 1
          (@nref $N A d->i_d+j_d) = $op((@nref $N A d->i_d+j_d), cols[r,c])
        end
      end
    end
    A
  end
end
