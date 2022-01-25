# forward declaration of macros used by library wrappers

# XXX: is it worth keeping this a macro, or should it just become a function?

macro context!(ex...)
    body = ex[end]
    ctx = ex[end-1]
    kwargs = ex[1:end-2]

    skip_destroyed = false
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        key, val = kwarg.args
        isa(key, Symbol) || throw(ArgumentError("non-symbolic keyword '$key'"))

        if key == :skip_destroyed
            skip_destroyed = val
        else
            throw(ArgumentError("unrecognized keyword argument '$kwarg'"))
        end
    end

    quote
        ctx = $(esc(ctx))
        if isvalid(ctx)
            old_ctx = context!(ctx)
            try
                $(esc(body))
            finally
                if old_ctx !== nothing && old_ctx != ctx && isvalid(old_ctx)
                    context!(old_ctx)
                end
            end
        elseif !$(esc(skip_destroyed))
            error("Cannot switch to an invalidated context.")
        end
    end
end
