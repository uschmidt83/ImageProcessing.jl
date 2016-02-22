module ImageProcessing

using Images
using Base.Cartesian

include("blockproc.jl")
include("filtering.jl")

export
  im2col,
  im2col!,
  imfiltermtx

end
