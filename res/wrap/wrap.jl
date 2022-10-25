# script to parse CUDA headers and generate Julia wrappers

using Clang
using Clang.Generators

using JuliaFormatter

using CUDA_full_jll, CUDNN_jll, CUTENSOR_jll, cuQuantum_jll
using Libglvnd_jll

function wrap(name, headers; targets=headers, defines=[], include_dirs=[])
    @info "Wrapping $name"

    args = get_default_args()
    append!(args, map(dir->"-I$dir", include_dirs))
    for define in defines
        if isa(define, Pair)
            append!(args, ["-D", "$(first(define))=$(last(define))"])
        else
            append!(args, ["-D", "$define"])
        end
    end

    options = load_options(joinpath(@__DIR__, "$(name).toml"))

    # create context
    ctx = create_context([headers...], args, options)

    # run generator
    build!(ctx, BUILDSTAGE_NO_PRINTING)

    # only keep the wrapped headers
    # NOTE: normally we'd do this by using `-isystem` instead of `-I` above,
    #       but in the case of CUDA most headers are in a single directory.
    replace!(get_nodes(ctx.dag)) do node
        path = normpath(Clang.get_filename(node.cursor))
        should_wrap = any(targets) do target
            occursin(target, path)
        end
        if !should_wrap
            return ExprNode(node.id, Generators.Skip(), node.cursor, Expr[], node.adj)
        end
        return node
    end

    rewriter!(ctx, options)

    build!(ctx, BUILDSTAGE_PRINTING_ONLY)

    format_file(options["general"]["output_file_path"], YASStyle())

    return
end

cuda_version_alias(lhs, rhs) = occursin(Regex("$(lhs)_v\\d"), rhs)
nvtx_string_alias(lhs, rhs) = occursin(Regex("$(lhs)(A|W|Ex)"), rhs)

function rewriter!(ctx, options)
    for node in get_nodes(ctx.dag)
        # remove aliases that are used to version functions
        #
        # when NVIDIA changes the behavior of an API, they version the function
        # (`cuFunction_v2`). To maintain backwards compatibility, they ship aliases with
        # their headers such that compiled binaries will keep using the old version, and
        # newly-compiled ones will use the developer's CUDA version. remove those, since we
        # target multiple CUDA versions.
        #
        # remove this if we ever decide to support a single supported version of CUDA.
        if node isa ExprNode{<:AbstractMacroNodeType}
            isempty(node.exprs) && continue
            expr = node.exprs[1]
            if Meta.isexpr(expr, :const)
                expr = expr.args[1]
            end
            if Meta.isexpr(expr, :(=))
                lhs, rhs = expr.args
                if Meta.isexpr(rhs, :call) && rhs.args[1] in (:__CUDA_API_PTDS, :__CUDA_API_PTSZ)
                    rhs = rhs.args[2]
                end
                if isa(rhs, Symbol) && (cuda_version_alias(String(lhs), String(rhs)) ||
                                        nvtx_string_alias(String(lhs), String(rhs)))
                    @debug "Removing version alias: `$expr`"
                    empty!(node.exprs)
                end
            end
        end

        if Generators.is_function(node) && !Generators.is_variadic_function(node)
            expr = node.exprs[1]
            call_expr = expr.args[2].args[1].args[3]    # assumes `@ccall`

            target_expr = call_expr.args[1].args[1]
            fn = String(target_expr.args[2].value)

            # rewrite pointer argument types
            arg_exprs = call_expr.args[1].args[2:end]
            if haskey(options, "api") && haskey(options["api"], fn)
                argtypes = get(options["api"][fn], "argtypes", Dict())
                for (arg, typ) in argtypes
                    i = parse(Int, arg)
                    arg_exprs[i].args[2] = Meta.parse(typ)
                end
            end

            # insert `initialize_context()` before each function with a `ccall`
            fn_options = if haskey(options, "api")
                get(options["api"], fn, Dict())
            else
                Dict{String,Any}()
            end
            if get(fn_options, "needs_context", true)
                pushfirst!(expr.args[2].args, :(initialize_context()))
            end

            # insert `@checked` before each function with a `ccall` returning a checked type`
            rettyp = call_expr.args[2]
            checked_types = if haskey(options, "api")
                get(options["api"], "checked_rettypes", Dict())
            else
                String[]
            end
            if rettyp isa Symbol && String(rettyp) in checked_types
                node.exprs[1] = Expr(:macrocall, Symbol("@checked"), nothing, expr)
            end
        end
    end
end

function main(name="all")
    cuda = joinpath(CUDA_full_jll.artifact_dir, "cuda", "include")
    cupti = joinpath(CUDA_full_jll.artifact_dir, "cuda", "extras", "CUPTI", "include")
    cudnn = joinpath(CUDNN_jll.artifact_dir, "include")
    cutensor = joinpath(CUTENSOR_jll.artifact_dir, "include")
    cuquantum = joinpath(cuQuantum_jll.artifact_dir, "include")
    opengl = joinpath(Libglvnd_jll.artifact_dir, "include")

    if name == "all" || name == "cudadrv"
        wrap("cuda", ["$cuda/cuda.h","$cuda/cudaGL.h","$cuda/cudaProfiler.h"];
            include_dirs=[cuda, opengl])
    end

    if name == "all" || name == "nvtx"
        wrap("nvtx", ["$cuda/nvtx3/nvToolsExt.h", "$cuda/nvtx3/nvToolsExtCuda.h"];
            include_dirs=[cuda])
    end

    if name == "all" || name == "nvml"
        wrap("nvml", ["$cuda/nvml.h"]; include_dirs=[cuda])
    end

    if name == "all" || name == "cupti"
        # NOTE: libclang (the C API) doesn't support/expose the __packed__/aligned attributes,
        #       so disable them (Julia doesn't support packed structs anyway)
        wrap("cupti", ["$cupti/cupti.h", "$cupti/cupti_profiler_target.h"];
            include_dirs=[cuda, cupti],
            targets=[r"cupti_.*.h"],
            defines=["__packed__"=>"", "aligned"=>""])
    end

    if name == "all" || name == "cublas"
        wrap("cublas", ["$cuda/cublas_v2.h", "$cuda/cublasXt.h"];
            targets=[r"cublas.*.h"],
            include_dirs=[cuda])
    end


    if name == "all" || name == "cufft"
        wrap("cufft", ["$cuda/cufft.h"]; include_dirs=[cuda])
    end

    if name == "all" || name == "curand"
        wrap("curand", ["$cuda/curand.h"]; include_dirs=[cuda])
    end

    if name == "all" || name == "cusparse"
        wrap("cusparse", ["$cuda/cusparse.h"]; include_dirs=[cuda])
    end

    if name == "all" || name == "cusolver"
        wrap("cusolver",
            ["$cuda/cusolverDn.h", "$cuda/cusolverSp.h",
             "$cuda/cusolverSp_LOWLEVEL_PREVIEW.h"];
            targets=[r"cusolver.*.h"],
            include_dirs=[cuda])

        wrap("cusolverRF", ["$cuda/cusolverRf.h"]; include_dirs=[cuda])

        wrap("cusolverMg", ["$cuda/cusolverMg.h"]; include_dirs=[cuda])
    end

    if name == "all" || name == "cudnn"
        wrap("cudnn",
            ["$cudnn/cudnn_version.h", "$cudnn/cudnn_ops_infer.h",
             "$cudnn/cudnn_ops_train.h", "$cudnn/cudnn_adv_infer.h",
             "$cudnn/cudnn_adv_train.h", "$cudnn/cudnn_cnn_infer.h",
             "$cudnn/cudnn_cnn_train.h"];
             include_dirs=[cuda, cudnn])
    end

    if name == "all" || name == "cutensor"
        wrap("cutensor", ["$cutensor/cutensor.h"];
            targets=["cutensor.h", "cutensor/types.h"],
            include_dirs=[cuda, cutensor])
    end

    if name == "all" || name == "cutensornet"
        wrap("cutensornet", ["$cuquantum/cutensornet.h"];
            targets=["cutensornet.h", "cutensornet/types.h"],
            include_dirs=[cuda, cuquantum])
    end

    if name == "all" || name == "custatevec"
        wrap("custatevec", ["$cuquantum/custatevec.h"];
            targets=["custatevec.h", "custatevec/types.h"],
            include_dirs=[cuda, cuquantum])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
