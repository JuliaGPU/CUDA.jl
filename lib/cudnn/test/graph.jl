using cuDNN:
    Graph,
    assign_uids!,
    conv_dgrad!,
    conv_fprop!,
    conv_wgrad!,
    CUDNN_DATA_INT8,
    is_supported,
    matmul!,
    norm_bwd!,
    norm_fwd!,
    pointwise!,
    reduction!,
    resample_bwd!,
    resample_fwd!,
    sdpa_bwd!,
    scalar!,
    sdpa_fwd!,
    tensor!,
    validate!

g = Graph(io_dtype=Float16)
x = tensor!(g; dims=(2, 3, 4), dtype=Float16, name="x")
@test x.backend_order == [3, 2, 1]
@test x.strides == [1, 2, 6]

q = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="Q")
k = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="K")
v = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="V")
scale = scalar!(g, Float32; rank=4, name="Scale")
o = sdpa_fwd!(g, q, k, v; scale)
@test o.dims == [64, 4, 32, 2]
@test q.backend_order == [4, 2, 3, 1]
@test o.backend_order == [4, 2, 3, 1]

oc = sdpa_fwd!(g, q, k, v; scale, causal=true)
@test oc.dims == [64, 4, 32, 2]
score = only(filter(t -> t.name == "Score", g.tensors))
@test score.dims == [32, 32, 4, 2]
@test score.backend_order == [4, 3, 2, 1]
@test only(filter(t -> t.name == "MaskValue", g.tensors)).by_value

seqq = tensor!(g; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenQ")
seqkv = tensor!(g; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenKV")
opad = sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqq, seq_len_kv=seqkv)
@test opad.dims == q.dims
@test last(g.ops).seq_len_q === seqq
@test_throws ArgumentError sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqq)
seqbad = tensor!(g; dims=(1, 1, 1, 3), dtype=Int32, name="BadSeqLen")
@test_throws DimensionMismatch sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqq,
                                         seq_len_kv=seqbad)
seqfloat = tensor!(g; dims=(1, 1, 1, 2), dtype=Float32, name="FloatSeqLen")
@test_throws ArgumentError sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqfloat,
                                     seq_len_kv=seqkv)

dO = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="dO")
stats = tensor!(g; dims=(1, 4, 32, 2), dtype=Float32, name="Stats")
dq, dk, dv = sdpa_bwd!(g, q, k, v, o, dO, stats; scale)
@test dq.dims == q.dims
@test dk.dims == k.dims
@test dv.dims == v.dims
@test dq.backend_order == [4, 2, 3, 1]
dqc, dkc, dvc = sdpa_bwd!(g, q, k, v, o, dO, stats; scale, causal=true)
@test dqc.dims == q.dims
@test dkc.dims == k.dims
@test dvc.dims == v.dims
@test last(g.ops).mask_subgraph !== nothing
dqp, dkp, dvp = sdpa_bwd!(g, q, k, v, o, dO, stats; scale, seq_len_q=seqq,
                           seq_len_kv=seqkv)
@test dqp.dims == q.dims
@test dkp.dims == k.dims
@test dvp.dims == v.dims
@test last(g.ops).seq_len_kv === seqkv

a = tensor!(g; dims=(8, 16, 2), dtype=Float16, name="A")
b = tensor!(g; dims=(16, 32, 2), dtype=Float16, name="B")
c = matmul!(g, a, b)
@test c.dims == [8, 32, 2]
b4 = tensor!(g; dims=(16, 32, 2, 2), dtype=Float16, name="B4")
@test_throws DimensionMismatch matmul!(g, a, b4)
bbad = tensor!(g; dims=(16, 32, 3), dtype=Float16, name="BBad")
@test_throws DimensionMismatch matmul!(g, a, bbad)

bias = tensor!(g; dims=(1, 32, 1), dtype=Float16, name="Bias")
y = pointwise!(g, :add, c, bias)
@test y.dims == [8, 32, 2]

r = reduction!(g, :sum, y; dims=1)
@test r.dims == [1, 32, 2]

cx = tensor!(g; dims=(8, 7, 3, 2), dtype=Float16, name="ConvX")
cw = tensor!(g; dims=(3, 2, 3, 5), dtype=Float16, name="ConvW")
cy = conv_fprop!(g, cx, cw; pre_padding=(1, 0), post_padding=(2, 1),
                 stride=(2, 1), dilation=(1, 2))
@test cy.dims == [5, 6, 5, 2]
cdx = conv_dgrad!(g, cy, cw; x_dims=cx.dims, pre_padding=(1, 0),
                  post_padding=(2, 1), stride=(2, 1), dilation=(1, 2))
@test cdx.dims == cx.dims
cdw = conv_wgrad!(g, cy, cx; w_dims=cw.dims, pre_padding=(1, 0),
                  post_padding=(2, 1), stride=(2, 1), dilation=(1, 2))
@test cdw.dims == cw.dims
gcx = tensor!(g; dims=(8, 7, 4, 2), dtype=Float16, name="GroupedConvX")
gcw = tensor!(g; dims=(3, 2, 2, 6), dtype=Float16, name="GroupedConvW")
@test conv_fprop!(g, gcx, gcw).dims == [6, 6, 6, 2]
badgcw = tensor!(g; dims=(3, 2, 2, 5), dtype=Float16, name="BadGroupedConvW")
@test_throws DimensionMismatch conv_fprop!(g, gcx, badgcw)
@test_throws ArgumentError conv_dgrad!(g, cy, cw; pre_padding=(1, 0),
                                       post_padding=(2, 1), stride=(2, 1),
                                       dilation=(1, 2))
@test_throws ArgumentError conv_wgrad!(g, cy, cx; pre_padding=(1, 0),
                                       post_padding=(2, 1), stride=(2, 1),
                                       dilation=(1, 2))

rx = tensor!(g; dims=(7, 6, 3, 2), dtype=Float16, name="PoolX")
ry = resample_fwd!(g, rx; mode=:avgpool_exclude_padding, window=(2, 3),
                   pre_padding=(1, 0), post_padding=(0, 1), stride=(2, 1))
@test ry.dims == [4, 5, 3, 2]
ryi, ridx = resample_fwd!(g, rx; mode=:maxpool, window=(2, 3),
                          pre_padding=(1, 0), post_padding=(0, 1),
                          stride=(2, 1), generate_index=true)
@test ryi.dims == ry.dims
@test ridx.dims == ry.dims
@test ridx.dtype == CUDNN_DATA_INT8
rdx = resample_bwd!(g, ryi; x_dims=rx.dims, index=ridx, mode=:maxpool,
                    window=(2, 3), pre_padding=(1, 0), post_padding=(0, 1),
                    stride=(2, 1))
@test rdx.dims == rx.dims
@test_throws ArgumentError resample_bwd!(g, ryi; mode=:maxpool, window=(2, 3),
                                         pre_padding=(1, 0), post_padding=(0, 1),
                                         stride=(2, 1))

nx = tensor!(g; dims=(7, 6, 3, 2), dtype=Float32, name="NormX")
nscale = tensor!(g; dims=(1, 1, 3, 1), dtype=Float32, name="NormScale")
nbias = tensor!(g; dims=(1, 1, 3, 1), dtype=Float32, name="NormBias")
neps = scalar!(g, Float32; rank=4, name="NormEpsilon")
ny, nmean, ninv, _, _ = norm_fwd!(g, nx, nscale, nbias; epsilon=neps)
@test ny.dims == nx.dims
@test nmean.dims == [1, 1, 3, 1]
@test ninv.dims == [1, 1, 3, 1]
ndx, ndscale, ndbias = norm_bwd!(g, ny, nx, nscale, nmean, ninv)
@test ndx.dims == nx.dims
@test ndscale.dims == nscale.dims
@test ndbias.dims == nbias.dims
badscale = tensor!(g; dims=(1, 1, 4, 1), dtype=Float32, name="BadScale")
@test_throws DimensionMismatch norm_fwd!(g, nx, badscale, nbias; epsilon=neps)

ghalf = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
hx = tensor!(ghalf; dims=(7, 6, 3, 2), dtype=Float16, name="X")
hscale = tensor!(ghalf; dims=(1, 1, 3, 1), dtype=Float32, name="Scale")
hbias = tensor!(ghalf; dims=(1, 1, 3, 1), dtype=Float32, name="Bias")
hy, hmean, hinv, _, _ = norm_fwd!(ghalf, hx, hscale, hbias)
@test hy.dtype == cuDNN.CUDNN_DATA_HALF
@test hmean.dtype == cuDNN.CUDNN_DATA_FLOAT
@test hinv.dtype == cuDNN.CUDNN_DATA_FLOAT
hscale16 = tensor!(ghalf; dims=(1, 1, 3, 1), dtype=Float16, name="Scale16")
@test_throws ArgumentError norm_fwd!(ghalf, hx, hscale16, hbias)

validate!(g)
assign_uids!(g)
@test all(!=(0), getfield.(g.tensors, :uid))
@test length(unique(getfield.(g.tensors, :uid))) == length(g.tensors)

gdup = Graph()
tensor!(gdup; dims=(1, 1, 1, 1), dtype=Float32, uid=1)
tensor!(gdup; dims=(1, 1, 1, 1), dtype=Float32, uid=1)
@test_throws ArgumentError validate!(gdup)

gunsupported = Graph(io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32)
ux = tensor!(gunsupported; dims=(4, 5, 3, 2), dtype=Float32, name="X")
us = tensor!(gunsupported; dims=(1, 1, 3, 1), dtype=Float32, name="Scale")
ub = tensor!(gunsupported; dims=(1, 1, 3, 1), dtype=Float32, name="Bias")
ue = scalar!(gunsupported, Float32; rank=4, name="Epsilon")
uy, um, ui, _, _ = norm_fwd!(gunsupported, ux, us, ub; epsilon=ue)
cuDNN.output!(uy)
cuDNN.output!(um)
cuDNN.output!(ui)
@test is_supported(gunsupported) isa Bool
