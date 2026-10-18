# Only source-identity provenance differs from the raw historical training
# oracle. Numerical expectations and the frozen file itself remain unchanged.
const TRAINING_PROVENANCE_OLD = "@test sort([name for name in keys(source_maps[1]) if source_maps[1][name]!=source_maps[2][name]])==[\"DstIntervalAdapter.jl\",\"DstIntervalObserver.jl\"]"
const TRAINING_PROVENANCE_NEW = "@test sort([name for name in keys(source_maps[1]) if source_maps[1][name]!=source_maps[2][name]])==[\"DstIntervalAdapter.jl\",\"DstIntervalObserver.jl\",\"DstIntervalStream.jl\",\"DstIntervalTraining.jl\"]"
const TRAINING_MODEL_SOURCE_NAMES = ("DstIntervalObserver.jl", "DstIntervalReplay.jl",
    "sandu_fullchannel_features.jl", "DstIntervalAdapter.jl", "DstIntervalStream.jl", "DstIntervalTraining.jl")
const TRAINING_EXPECTED_CHANGED_SOURCES = ["DstIntervalAdapter.jl", "DstIntervalObserver.jl",
    "DstIntervalStream.jl", "DstIntervalTraining.jl"]

function provenance_ast(node)
    node isa Expr || return node
    Expr(node.head, (provenance_ast(child) for child in node.args if !(child isa LineNumberNode))...)
end

function rewrite_training_provenance(node, count)
    if node isa Expr && isequal(provenance_ast(node), provenance_ast(Meta.parse(TRAINING_PROVENANCE_OLD)))
        count[] += 1
        replacement = Meta.parse(TRAINING_PROVENANCE_NEW)
        # Retain the original macro call's location while changing only the
        # reviewed right-hand literal of this exact provenance assertion.
        replacement.args[2] = node.args[2]
        return replacement
    end
    node isa Expr || return node
    Expr(node.head, (rewrite_training_provenance(child, count) for child in node.args)...)
end

function prepared_training_provenance(code)
    count = Ref(0)
    mapped = rewrite_training_provenance(Meta.parseall(code), count)
    count[] == 1 || throw(ArgumentError("exactly one frozen training provenance assertion required"))
    mapped
end

function evaluate_training_provenance(target, path, marker, prefix, source_pins, baseline_pins, model_sources)
    Set(model_sources) == Set(TRAINING_MODEL_SOURCE_NAMES) || error("model source-name contract changed")
    changed = sort([name for name in TRAINING_MODEL_SOURCE_NAMES if source_pins[name] != baseline_pins[name]])
    changed == TRAINING_EXPECTED_CHANGED_SOURCES || error("literal source maps disagree with reviewed provenance delta")
    parts = split(read(path, String), marker; limit=2)
    length(parts) == 2 || error("frozen training binding marker absent")
    code = prefix * marker * parts[2]
    prepared_training_provenance(code) # fail before executing missing/ambiguous context
    count = Ref(0)
    Base.include_string(node -> rewrite_training_provenance(node, count), target, code, path)
    count[] == 1 || error("training provenance transformation count changed")
end

function test_training_provenance_binding()
    @testset "Exact source-provenance AST binding" begin
        expected = Meta.parseall(TRAINING_PROVENANCE_NEW)
        @test isequal(provenance_ast(prepared_training_provenance(TRAINING_PROVENANCE_OLD)), provenance_ast(expected))
        prefix = "@test reinterpret(UInt64, 1.0) == 0x3ff0000000000000\n"
        suffix = "\n@test isapprox(1., 1.; atol=1e-12, rtol=0.)\n"
        @test isequal(provenance_ast(prepared_training_provenance(prefix * TRAINING_PROVENANCE_OLD * suffix)),
            provenance_ast(Meta.parseall(prefix * TRAINING_PROVENANCE_NEW * suffix)))
        for mutation in ("", TRAINING_PROVENANCE_NEW,
            TRAINING_PROVENANCE_OLD * "\n" * TRAINING_PROVENANCE_OLD,
            replace(TRAINING_PROVENANCE_OLD, "sort(" => "reverse("),
            replace(TRAINING_PROVENANCE_OLD, "DstIntervalObserver.jl" => "DstIntervalReplay.jl"),
            replace(TRAINING_PROVENANCE_OLD, "==[" => "!=["))
            @test_throws ArgumentError prepared_training_provenance(mutation)
        end
    end
end
