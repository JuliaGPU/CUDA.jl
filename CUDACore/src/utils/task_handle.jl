# per-task library handles, borrowed from a `HandleCache`

"""
    TaskHandleState{W,S}

What [`task_handle`](@ref) keeps per task and key: the handle wrapper and the settings it
was last configured with. Use it as the value type of the `TaskLocalCache` passed to
`task_handle`, e.g. `TaskLocalCache{CuContext, TaskHandleState{Handle, Tuple{CuStream}}}`.
"""
const TaskHandleState{W,S} = @NamedTuple{handle::W, settings::S}

"""
    task_handle(wrap, release, states::TaskLocalCache, cache::HandleCache, key,
                settings=(), setters=()) -> wrapper

Return the current task's library handle for `key` (typically the active context).

The first call in a task takes a handle from `cache` with `pop!(cache, key)`, wraps it with
`wrap(handle)` in a mutable object that the task keeps in `states`, and registers
`release(wrapper)` with [`handle_finalizer`](@ref) to put the handle back once the wrapper
is collected or the task has finished. `wrap` may allocate memory that lives as long as the
handle (e.g. a workspace), and `release` must free it before putting the handle back.

`settings` is a tuple of the task state the handle depends on, such as the active stream,
and `setters` a tuple of functions that apply them: `setters[i](wrapper, settings[i])` is
called when the handle is created, and whenever `settings[i]` differs from the value the
handle was last configured with.
"""
@inline function task_handle(wrap::W, release::R,
                             states::TaskLocalCache{K,TaskHandleState{H,S}},
                             cache::HandleCache{K}, key::K,
                             settings::S=(), setters::T=()) where {W,R,K,H,S,T}
    @noinline function new_state()
        handle = pop!(cache, key)
        wrapper = wrap(handle)
        handle_finalizer(release, wrapper, cache, key, handle)
        foreach((set, value) -> set(wrapper, value), setters, settings)
        (; handle=wrapper, settings)
    end
    state = get!(new_state, states, key)

    @noinline function update_settings(wrapper, previous)
        foreach(setters, settings, previous) do set, value, old
            value == old || set(wrapper, value)
        end
        task_dict(states)[key] = (; handle=wrapper, settings)
        wrapper
    end
    if state.settings != settings
        return update_settings(state.handle, state.settings)::H
    end

    return state.handle
end
