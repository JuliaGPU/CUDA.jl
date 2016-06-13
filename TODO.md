* Merge with pointers in CUDArt.jl

* Set inner pointers to null upon free, and use Base.cconvert to check before usage
  cfr. https://github.com/yuyichao/LibArchive.jl/blob/83961ef63f9cd6eddf7ecaf2f7f93eda16429009/src/entry.jl#L58