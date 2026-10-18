# Literal independent module-load expectations, frozen before parser edits.
# Fixture sources are strings and are parsed, never executed.
const CASES = [
    (name="comma", expected=["Dates","SHA","TOML"], malformed=false, source="using Dates,SHA,TOML\n"),
    (name="multiline", expected=["Dates","SHA","TOML"], malformed=false, source="using Dates,\n    SHA,\n    TOML\n"),
    (name="semicolon", expected=["Dates","SHA","TOML"], malformed=false, source="using Dates; import SHA, TOML\n"),
    (name="qualified", expected=["LinearAlgebra","Base","SHA"], malformed=false, source="using LinearAlgebra.BLAS, Base.Threads\nimport SHA.sha256\n"),
    (name="selective", expected=["Dates","SHA"], malformed=false, source="using Dates: DateTime, Hour\nimport SHA: sha256, sha1\n"),
    (name="import_alias", expected=["Dates","SHA"], malformed=false, source="import Dates as D\nimport SHA.sha256 as digest\n"),
    (name="selective_alias", expected=["SHA"], malformed=false, source="import SHA: sha256 as digest, sha1\n"),
    (name="relative", expected=String[], malformed=false, source="using .Local, ..Parent.Child\nimport .Dates: Date\n"),
    (name="mixed_relative", expected=["Dates","SHA"], malformed=false, source="using .Local, Dates\nimport ..Parent, SHA\n"),
    (name="comments_strings", expected=["Dates"], malformed=false, source="# using CommentPkg\n#=\nusing BlockCommentPkg\n=#\nconst TEXT = \"\"\"\nusing StringPkg\n\"\"\"\nusing Dates #, SHA\n"),
    (name="quoted", expected=["Dates"], malformed=false, source="quote\n using QuotedPkg\nend\nconst expression = :(import QuoteExprPkg)\nusing Dates\n"),
    (name="quoted_module", expected=["SHA"], malformed=false, source=":(module Fiction\n using GhostPkg\nend)\nimport SHA\n"),
    (name="nested_module", expected=["Dates","SHA"], malformed=false, source="module Inner\nusing Dates,SHA\nend\n"),
    (name="identifiers", expected=["Módulo","_PrivateRoot"], malformed=false, source="using Módulo, _PrivateRoot\n"),
    (name="malformed_comma", expected=String[], malformed=true, source="using Dates,\n"),
    (name="malformed_selective", expected=String[], malformed=true, source="using Dates:\n"),
    (name="malformed_quote", expected=String[], malformed=true, source="quote\nusing Dates\n"),
    (name="malformed_string", expected=String[], malformed=true, source="const s = \"hello"),
    (name="malformed_comment", expected=String[], malformed=true, source="#= using Dates\n"),
]
