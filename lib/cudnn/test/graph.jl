using cuDNN:
    Graph,
    build!,
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
    reshape!,
    sdpa_bwd!,
    scalar!,
    sdpa_fwd!,
    tensor!

g = Graph(io_dtype=Float16)
x = tensor!(g; dims=(2, 3, 4), dtype=Float16, name="x")
@test x.strides == [1, 2, 6]

q = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="Q")
k = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="K")
v = tensor!(g; dims=(64, 4, 32, 2), dtype=Float16, name="V")
scale = scalar!(g, Float32; rank=4, name="Scale")
o = sdpa_fwd!(g, q, k, v; scale)
@test o.dims == [64, 4, 32, 2]

oc = sdpa_fwd!(g, q, k, v; scale, causal=true)
@test oc.dims == [64, 4, 32, 2]

# without padding, bottom-right alignment takes the lengths as by-value scalars
gbr = Graph(io_dtype=Float16)
brq = tensor!(gbr; dims=(64, 4, 16, 2), dtype=Float16, name="Q")
brkv = tensor!(gbr; dims=(64, 4, 32, 2), dtype=Float16, name="KV")
sdpa_fwd!(gbr, brq, brkv, brkv; causal=:bottom_right)
@test cuDNN.tensor(gbr, "MaskSeqLenQ").by_value
@test cuDNN.tensor(gbr, "MaskSeqLenKV").dtype == cuDNN.CUDNN_DATA_INT32
@test_throws cuDNN.UnsupportedGraphError sdpa_fwd!(gbr, brkv, brq, brq;
                                                  causal=:bottom_right)
@test_throws ArgumentError sdpa_fwd!(gbr, brq, brkv, brkv; causal=:diagonal)

# diagonal-band bounds are Int32 scalars; a causal mask already sets the right bound
gband = Graph(io_dtype=Float16)
lb = scalar!(gband, Int32; rank=4, name="LeftBound")
rb = scalar!(gband, Int32; rank=4, name="RightBound")
bq = tensor!(gband; dims=(64, 4, 32, 2), dtype=Float16, name="Q")
sdpa_fwd!(gband, bq, bq, bq; causal=true, left_bound=lb)
sdpa_fwd!(gband, bq, bq, bq; left_bound=lb, right_bound=rb)
@test_throws ArgumentError sdpa_fwd!(gband, bq, bq, bq; causal=true, right_bound=rb)
lbf = scalar!(gband, Float32; rank=4, name="FloatLeftBound")
@test_throws ArgumentError sdpa_fwd!(gband, bq, bq, bq; left_bound=lbf)
bkv = tensor!(gband; dims=(64, 4, 16, 2), dtype=Float16, name="KV")
@test_throws cuDNN.UnsupportedGraphError sdpa_fwd!(gband, bq, bkv, bkv; left_bound=lb)

seqq = tensor!(g; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenQ")
seqkv = tensor!(g; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenKV")
opad = sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqq, seq_len_kv=seqkv)
@test opad.dims == q.dims
@test_throws ArgumentError sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqq)
seqbad = tensor!(g; dims=(1, 1, 1, 3), dtype=Int32, name="BadSeqLen")
@test_throws DimensionMismatch sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqq,
                                         seq_len_kv=seqbad)
seqfloat = tensor!(g; dims=(1, 1, 1, 2), dtype=Float32, name="FloatSeqLen")
@test_throws ArgumentError sdpa_fwd!(g, q, k, v; scale, seq_len_q=seqfloat,
                                     seq_len_kv=seqkv)

# the composite backward uses (d, s, h, b) dims, unlike the forward
gbwd = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
bq = tensor!(gbwd; dims=(64, 32, 4, 2), dtype=Float16, name="Q")
bk = tensor!(gbwd; dims=(64, 48, 4, 2), dtype=Float16, name="K")
bv = tensor!(gbwd; dims=(64, 48, 4, 2), dtype=Float16, name="V")
bo = tensor!(gbwd; dims=(64, 32, 4, 2), dtype=Float16, name="O")
bdO = tensor!(gbwd; dims=(64, 32, 4, 2), dtype=Float16, name="dO")
bstats = tensor!(gbwd; dims=(1, 32, 4, 2), dtype=Float32, name="Stats")
dq, dk, dv, ssum, dqacc = sdpa_bwd!(gbwd, bq, bk, bv, bo, bdO, bstats)
@test dq.dims == [64, 32, 4, 2]
@test dk.dims == [64, 48, 4, 2]
@test dv.dims == [64, 48, 4, 2]
@test ssum.dims == [1, 32, 4, 2]
@test ssum.dtype == cuDNN.CUDNN_DATA_FLOAT
@test dqacc.dims == [64, 32, 4, 2]
@test dqacc.dtype == cuDNN.CUDNN_DATA_FLOAT
# k and v were re-presented in place as their transposes
@test bk.dims == [48, 64, 4, 2]
@test bk.strides == [64, 1, 64 * 48, 64 * 48 * 4]

gbwdc = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
cq = tensor!(gbwdc; dims=(64, 32, 4, 2), dtype=Float16, name="Q")
ck = tensor!(gbwdc; dims=(64, 48, 4, 2), dtype=Float16, name="K")
cv = tensor!(gbwdc; dims=(64, 48, 4, 2), dtype=Float16, name="V")
co = tensor!(gbwdc; dims=(64, 32, 4, 2), dtype=Float16, name="O")
cdO = tensor!(gbwdc; dims=(64, 32, 4, 2), dtype=Float16, name="dO")
cstats = tensor!(gbwdc; dims=(1, 32, 4, 2), dtype=Float32, name="Stats")
cdq, _ = sdpa_bwd!(gbwdc, cq, ck, cv, co, cdO, cstats; causal=true)
@test cdq.dims == [64, 32, 4, 2]
@test cuDNN.tensor(gbwdc, "MaskValue").by_value

gbwdbr = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
brq = tensor!(gbwdbr; dims=(64, 32, 4, 2), dtype=Float16, name="Q")
brk = tensor!(gbwdbr; dims=(64, 48, 4, 2), dtype=Float16, name="K")
brv = tensor!(gbwdbr; dims=(64, 48, 4, 2), dtype=Float16, name="V")
bro = tensor!(gbwdbr; dims=(64, 32, 4, 2), dtype=Float16, name="O")
brdO = tensor!(gbwdbr; dims=(64, 32, 4, 2), dtype=Float16, name="dO")
brstats = tensor!(gbwdbr; dims=(1, 32, 4, 2), dtype=Float32, name="Stats")
sdpa_bwd!(gbwdbr, brq, brk, brv, bro, brdO, brstats; causal=:bottom_right)
@test cuDNN.tensor(gbwdbr, "MaskSeqLenQ").by_value
@test cuDNN.tensor(gbwdbr, "MaskSeqLenKV").by_value

# a deterministic backward writes dQ directly, without the Float32 accumulator
let g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    io = (tensor!(g; dims=(64, 32, 4, 2), dtype=Float16, name="Q"),
          tensor!(g; dims=(64, 48, 4, 2), dtype=Float16, name="K"),
          tensor!(g; dims=(64, 48, 4, 2), dtype=Float16, name="V"),
          tensor!(g; dims=(64, 32, 4, 2), dtype=Float16, name="O"),
          tensor!(g; dims=(64, 32, 4, 2), dtype=Float16, name="dO"),
          tensor!(g; dims=(1, 32, 4, 2), dtype=Float32, name="Stats"))
    acc = tensor!(g; dims=(64, 32, 4, 2), dtype=Float32, name="Accum")
    @test_throws ArgumentError sdpa_bwd!(g, io...; dQ_accum=acc, deterministic=true)
    @test sdpa_bwd!(g, io...; deterministic=true)[5] === nothing
    @test !any(t -> t.name == "dQAccum", g.tensors)
end

# grouped queries add fullhead workspaces folded onto the shared k/v heads
ggqa = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
mq = tensor!(ggqa; dims=(64, 32, 4, 2), dtype=Float16, name="Q")
mk = tensor!(ggqa; dims=(64, 48, 2, 2), dtype=Float16, name="K")
mv = tensor!(ggqa; dims=(64, 48, 2, 2), dtype=Float16, name="V")
mo = tensor!(ggqa; dims=(64, 32, 4, 2), dtype=Float16, name="O")
mdO = tensor!(ggqa; dims=(64, 32, 4, 2), dtype=Float16, name="dO")
mstats = tensor!(ggqa; dims=(1, 32, 4, 2), dtype=Float32, name="Stats")
mdq, mdk, mdv, _, _, mdkf, mdvf = sdpa_bwd!(ggqa, mq, mk, mv, mo, mdO, mstats)
@test mdq.dims == [64, 32, 4, 2]
@test mdk.dims == [64, 48, 2, 2]
@test mdv.dims == [64, 48, 2, 2]
@test mdkf.dims == [64, 48, 4, 2]
@test mdvf.dims == [64, 48, 4, 2]

# ragged tensors need padding masks; ragged gradients accumulate in ragged workspaces
function ragged_bwd_graph(; hk=2)
    g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    offq = tensor!(g; dims=(1, 1, 1, 3), dtype=Int64, name="OffQ")
    offkv = tensor!(g; dims=(1, 1, 1, 3), dtype=Int64, name="OffKV")
    ragged(name, s, nh, off) = tensor!(g; dims=(64, s, nh, 2), dtype=Float16, name,
                                       strides=(1, 64nh, 64, 64nh), ragged_offset=off)
    io = (ragged("Q", 32, 4, offq), ragged("K", 48, hk, offkv), ragged("V", 48, hk, offkv),
          ragged("O", 32, 4, offq), ragged("dO", 32, 4, offq),
          tensor!(g; dims=(1, 32, 4, 2), dtype=Float32, name="Stats"))
    grads = (; dQ=ragged("dQ", 32, 4, offq), dK=ragged("dK", 48, hk, offkv),
             dV=ragged("dV", 48, hk, offkv))
    lens = (; seq_len_q=tensor!(g; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenQ"),
            seq_len_kv=tensor!(g; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenKV"))
    return g, io, grads, lens
end
let (g, io, grads, lens) = ragged_bwd_graph()
    @test_throws ArgumentError sdpa_bwd!(g, io...; grads...)
end
let (g, io, grads, lens) = ragged_bwd_graph()
    _, _, _, rsum, racc, rdkf, rdvf = sdpa_bwd!(g, io...; grads..., lens...)
    @test rsum.ragged_offset === nothing
    @test racc.ragged_offset === grads.dQ.ragged_offset
    @test racc.strides == grads.dQ.strides
    # the full-head workspaces take offsets of their own, over 4 heads per token
    @test rdkf.ragged_offset === rdvf.ragged_offset ===
          cuDNN.tensor(g, "FullheadRaggedOffset")
    @test rdkf.strides == [1, 256, 64, 256]
end
let (g, io, grads, lens) = ragged_bwd_graph(; hk=1)
    rdkf = sdpa_bwd!(g, io...; grads..., lens...)[6]
    @test rdkf.strides == [1, 256, 64, 256]   # not the lone K head's canonical stride
end
let (g, io, grads, lens) = ragged_bwd_graph()
    rdq, rdk = sdpa_bwd!(g, io...; lens...)   # default gradients follow ragged inputs
    @test rdq.ragged_offset === io[1].ragged_offset
    @test rdq.strides == io[1].strides
    @test rdk.ragged_offset === io[2].ragged_offset
end
let (g, io, grads, lens) = ragged_bwd_graph()
    acc = tensor!(g; dims=(64, 32, 4, 2), strides=(1, 64, 2048, 8192), dtype=Float32,
                  name="dQAccum", ragged_offset=grads.dQ.ragged_offset)
    @test_throws ArgumentError sdpa_bwd!(g, io...; grads..., lens..., dQ_accum=acc)
end
let (g, io, grads, lens) = ragged_bwd_graph()
    stats = tensor!(g; dims=(1, 32, 4, 2), strides=(1, 4, 1, 128), dtype=Float32,
                    name="RaggedStats", ragged_offset=grads.dQ.ragged_offset)
    @test_throws ArgumentError sdpa_bwd!(g, io[1:5]..., stats; grads..., lens...)
end

# a ragged binding is checked within a token only: the offsets place the tokens
let g = Graph(io_dtype=Float16)
    off = tensor!(g; dims=(1, 1, 1, 3), dtype=Int64, name="Off")
    t = tensor!(g; dims=(64, 4, 32, 2), strides=(1, 64, 256, 256), dtype=Float16,
                name="Q", ragged_offset=off)
    @test cuDNN.checked_array_pointer(t, CUDACore.zeros(Float16, 64, 4, 77)) != C_NULL
    @test_throws DimensionMismatch cuDNN.checked_array_pointer(
        t, CUDACore.zeros(Float16, 64, 2, 77))
    @test_throws DimensionMismatch cuDNN.checked_array_pointer(
        t, CUDACore.zeros(Float16, 256, 77))
    @test_throws ArgumentError sdpa_fwd!(g, t, t, t)    # without padding masks
end

# io strides are free: a (d, h, s, b)-packed buffer binds through transposed strides
gbshd = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
tq = tensor!(gbshd; dims=(64, 32, 4, 2), strides=(1, 256, 64, 8192),
              dtype=Float16, name="Q")
tk = tensor!(gbshd; dims=(64, 48, 4, 2), strides=(1, 256, 64, 12288),
             dtype=Float16, name="K")
tv = tensor!(gbshd; dims=(64, 48, 4, 2), strides=(1, 256, 64, 12288),
             dtype=Float16, name="V")
to = tensor!(gbshd; dims=(64, 32, 4, 2), strides=(1, 256, 64, 8192),
             dtype=Float16, name="O")
tdO = tensor!(gbshd; dims=(64, 32, 4, 2), strides=(1, 256, 64, 8192),
              dtype=Float16, name="dO")
tstats = tensor!(gbshd; dims=(1, 32, 4, 2), dtype=Float32, name="Stats")
tdq, _ = sdpa_bwd!(gbshd, tq, tk, tv, to, tdO, tstats)
@test tdq.dims == [64, 32, 4, 2]
@test tk.strides == [256, 1, 64, 12288]

# rejections: unpaired seq lens, indivisible heads, head dim not innermost,
# non-packed stats/workspaces
gbad = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
xq = tensor!(gbad; dims=(64, 32, 4, 2), dtype=Float16, name="Q")
xk = tensor!(gbad; dims=(64, 48, 4, 2), dtype=Float16, name="K")
xv = tensor!(gbad; dims=(64, 48, 4, 2), dtype=Float16, name="V")
xo = tensor!(gbad; dims=(64, 32, 4, 2), dtype=Float16, name="O")
xdO = tensor!(gbad; dims=(64, 32, 4, 2), dtype=Float16, name="dO")
xstats = tensor!(gbad; dims=(1, 32, 4, 2), dtype=Float32, name="Stats")
xseqq = tensor!(gbad; dims=(1, 1, 1, 2), dtype=Int32, name="SeqLenQ")
@test_throws ArgumentError sdpa_bwd!(gbad, xq, xk, xv, xo, xdO, xstats;
                                     seq_len_q=xseqq)
kgqa = tensor!(gbad; dims=(64, 48, 3, 2), dtype=Float16, name="Kgqa")
vgqa = tensor!(gbad; dims=(64, 48, 3, 2), dtype=Float16, name="Vgqa")
@test_throws DimensionMismatch sdpa_bwd!(gbad, xq, kgqa, vgqa, xo, xdO, xstats)
qbad = tensor!(gbad; dims=(64, 32, 4, 2), strides=(32, 1, 2048, 8192),
               dtype=Float16, name="Qbad")
@test_throws ArgumentError sdpa_bwd!(gbad, qbad, xk, xv, xo, xdO, xstats)
badstats = tensor!(gbad; dims=(1, 32, 4, 2), strides=(1, 4, 1, 128),
                   dtype=Float32, name="BadStats")
@test_throws ArgumentError sdpa_bwd!(gbad, xq, xk, xv, xo, xdO, badstats)
badsum = tensor!(gbad; dims=(1, 32, 4, 2), strides=(1, 4, 1, 128),
                 dtype=Float32, name="BadSum")
@test_throws ArgumentError sdpa_bwd!(gbad, xq, xk, xv, xo, xdO, xstats;
                                     softmax_sum=badsum)

rsx = tensor!(g; dims=(8, 16, 2), dtype=Float16, name="ReshapeX")
rsy = reshape!(g, rsx; dims=(16, 8, 2), strides=(8, 1, 128), name="ReshapeY")
@test rsy.dims == [16, 8, 2]
@test rsy.virtual
@test_throws DimensionMismatch reshape!(g, rsx; dims=(4, 4, 2))

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
# layer/RMS norm: scale spans the normalized dims, stats span the complement
lscale = tensor!(g; dims=(7, 1, 1, 1), dtype=Float32, name="LayerScale")
lbias = tensor!(g; dims=(7, 1, 1, 1), dtype=Float32, name="LayerBias")
lny, lnmean, lninv, _, _ = norm_fwd!(g, nx, lscale, lbias; mode=:layernorm)
@test lny.dims == nx.dims
@test lnmean.dims == [1, 6, 3, 2]
@test lninv.dims == [1, 6, 3, 2]
rny, rnmean, rninv, _, _ = norm_fwd!(g, nx, lscale, nothing; mode=:rmsnorm)
@test rnmean === nothing
@test rninv.dims == [1, 6, 3, 2]
@test norm_fwd!(g, nx, lscale, nothing; mode=:rmsnorm, phase=:inference).dims == nx.dims
lbdx, lbdscale, lbdbias = norm_bwd!(g, lny, nx, lscale, lnmean, lninv; mode=:layernorm)
@test lbdx.dims == nx.dims
@test lbdscale.dims == [7, 1, 1, 1]
@test lbdbias.dims == [7, 1, 1, 1]
rbdx, rbdscale, rbdbias = norm_bwd!(g, rny, nx, lscale, nothing, rninv; mode=:rmsnorm)
@test rbdscale.dims == [7, 1, 1, 1]
@test rbdbias === nothing
@test_throws ArgumentError norm_bwd!(g, rny, nx, lscale, lnmean, rninv; mode=:rmsnorm)
@test_throws ArgumentError norm_bwd!(g, lny, nx, lscale, nothing, lninv; mode=:layernorm)
@test_throws ArgumentError norm_fwd!(g, nx, lscale, nothing; mode=:layernorm)
@test_throws ArgumentError norm_fwd!(g, nx, lscale, lbias; mode=:rmsnorm, mean=lnmean)
@test_throws ArgumentError norm_fwd!(g, nx, nscale, nbias;
                                     mode=cuDNN.CUDNN_GROUP_NORM)
@test_throws ArgumentError norm_fwd!(g, nx, nscale, nbias; phase=:inference,
                                     mean=nmean, inv_variance=ninv, epsilon=neps)
@test_throws ArgumentError norm_fwd!(g, nx, nscale, nbias; phase=:inference,
                                     mean=nmean, inv_variance=ninv,
                                     input_running_mean=nmean)

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

gdup = Graph()
tensor!(gdup; dims=(1, 1, 1, 1), dtype=Float32, uid=1)
tensor!(gdup; dims=(1, 1, 1, 1), dtype=Float32, uid=1)
@test_throws ArgumentError build!(gdup)

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
