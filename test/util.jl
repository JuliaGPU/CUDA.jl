macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))
                end
            end
            ret, read(fname, String)
        end
    end
end

# some tests are mysteriously broken with certain hardware/software.
# use a horrible macro to mark those tests as "potentially broken"
@eval Test begin
    export @test_maybe_broken

    macro test_maybe_broken(ex, kws...)
        test_expr!("@test_maybe_broken", ex, kws...)
        orig_ex = Expr(:inert, ex)
        result = get_test_result(ex, __source__)
        quote
            x = $result
            if x.value
                do_test(x, $orig_ex)
            else
                do_broken_test(x, $orig_ex)
            end
        end
    end
end
