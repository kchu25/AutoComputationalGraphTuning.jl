# ==============================================================================
# Verbosity control
#
# A single global verbosity level gates all console output. Levels are
# cumulative: a higher level shows everything the lower levels show plus more.
#
#   :silent  (0) - nothing (not even results)
#   :quiet   (1) - errors, warnings, and final results only            [DEFAULT]
#   :normal  (2) - + milestones (training start, trials, new-best, ...)
#   :verbose (3) - + per-epoch summaries and search progress
#   :debug   (4) - + per-batch loss
#
# Control it at runtime with `set_verbosity!(:verbose)` (or an Int 0:4).
# `Base.show` methods print to a caller-supplied IO and are intentionally
# NOT gated by this mechanism.
# ==============================================================================

const VERBOSITY_SILENT  = 0
const VERBOSITY_QUIET   = 1
const VERBOSITY_NORMAL  = 2
const VERBOSITY_VERBOSE = 3
const VERBOSITY_DEBUG   = 4

const _VERBOSITY_NAMES = Dict(
    :silent  => VERBOSITY_SILENT,
    :quiet   => VERBOSITY_QUIET,
    :normal  => VERBOSITY_NORMAL,
    :verbose => VERBOSITY_VERBOSE,
    :debug   => VERBOSITY_DEBUG,
)

# Default: quiet (results, errors and warnings only)
const _VERBOSITY = Ref(VERBOSITY_QUIET)

"""
    set_verbosity!(level)

Set the global console-output verbosity. `level` may be a `Symbol`
(`:silent`, `:quiet`, `:normal`, `:verbose`, `:debug`) or an `Integer` in `0:4`.
Returns the new level (as an `Int`).
"""
function set_verbosity!(level::Integer)
    (VERBOSITY_SILENT <= level <= VERBOSITY_DEBUG) ||
        throw(ArgumentError("verbosity level must be in 0:4, got $level"))
    _VERBOSITY[] = Int(level)
end

function set_verbosity!(level::Symbol)
    haskey(_VERBOSITY_NAMES, level) ||
        throw(ArgumentError("unknown verbosity :$level; choose from $(sort(collect(keys(_VERBOSITY_NAMES))))"))
    _VERBOSITY[] = _VERBOSITY_NAMES[level]
end

"""
    get_verbosity() -> Int

Return the current global verbosity level (`0:4`).
"""
get_verbosity() = _VERBOSITY[]

@inline _should_log(level::Integer) = _VERBOSITY[] >= level

"""
    vprintln(level, args...)

`println(args...)` only if the current verbosity is at least `level`.
"""
function vprintln(level::Integer, args...)
    _should_log(level) && println(args...)
    nothing
end

"""
    vprint(level, args...)

`print(args...)` only if the current verbosity is at least `level`.
"""
function vprint(level::Integer, args...)
    _should_log(level) && print(args...)
    nothing
end
