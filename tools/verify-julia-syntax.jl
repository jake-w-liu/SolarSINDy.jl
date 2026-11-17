"""Parse Julia source without evaluating candidates; reject embedded parser errors."""
module StrictJuliaSyntax

export has_parse_error, check_text, check_file, main

function has_parse_error(tree)::Bool
    tree isa Expr || return false
    tree.head in (:error, :incomplete) && return true
    return any(has_parse_error, tree.args)
end

function check_text(text::AbstractString; filename::AbstractString="none")
    tree = Meta.parseall(text; filename=filename)
    has_parse_error(tree) && throw(ArgumentError("malformed or incomplete Julia source: " * filename))
    return nothing
end

function check_file(path::AbstractString)
    check_text(read(path, String); filename=path)
    return nothing
end

function main(paths)::Int
    isempty(paths) && return 1
    failed = false
    for path in paths
        try
            check_file(path)
        catch exception
            exception isa InterruptException && rethrow()
            failed = true
            println(stderr, "Julia syntax check failed: ", path)
            showerror(stderr, exception)
            println(stderr)
        end
    end
    return failed ? 1 : 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(StrictJuliaSyntax.main(ARGS))
end
