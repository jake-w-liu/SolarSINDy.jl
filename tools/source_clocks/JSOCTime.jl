module JSOCTime

using Dates

export parse_trec_utc

# IERS Leap_Second.dat, updated through Bulletin C72 (July2026), expires
# 28June2027. This table is deliberately limited to the supported UTC interval.
# https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat
# https://datacenter.iers.org/data/html/bulletinc-072.html
const _UTC_FIRST = DateTime(2009, 1, 1)
const _UTC_STOP_EXCLUSIVE = DateTime(2027, 6, 28)
const _UTC_EFFECTIVE = (DateTime(2009, 1, 1), DateTime(2012, 7, 1),
                        DateTime(2015, 7, 1), DateTime(2017, 1, 1))
const _TAI_MINUS_UTC = (34, 35, 36, 37)
const _TAI_CUTOFFS = ntuple(i -> _UTC_EFFECTIVE[i] + Second(_TAI_MINUS_UTC[i]), 4)
const _TREC_PATTERN = r"\A([0-9]{4})\.([0-9]{2})\.([0-9]{2})_([0-9]{2}):([0-9]{2}):([0-9]{2})_TAI\z"

"""
    parse_trec_utc(label::AbstractString)::DateTime

Convert a strict whole-second JSOC `yyyy.mm.dd_HH:MM:SS_TAI` label to UTC.
The supported UTC range is [2009-01-01,2027-06-28). Select the offset at its
effective TAI cutoff, not by the label's calendar date. An inserted UTC leap
second has no Julia `DateTime` representation and is rejected explicitly.
"""
function parse_trec_utc(label::AbstractString)::DateTime
    matched = match(_TREC_PATTERN, label)
    matched === nothing && throw(ArgumentError("strict yyyy.mm.dd_HH:MM:SS_TAI label required"))
    yy, mm, dd, hh, mi, ss = ntuple(i -> parse(Int, matched.captures[i]), 6)
    0 <= hh <= 23 && 0 <= mi <= 59 && 0 <= ss <= 59 ||
        throw(ArgumentError("JSOC TAI clock fields are outside their valid ranges"))
    tai = DateTime(yy, mm, dd, hh, mi, ss)
    (year(tai), month(tai), day(tai), hour(tai), minute(tai), second(tai)) ==
        (yy, mm, dd, hh, mi, ss) || throw(ArgumentError("JSOC TAI date must not be normalized"))
    # Each supported transition inserts exactly one UTC second immediately
    # before the new offset's effective TAI cutoff.
    for i in 2:length(_TAI_CUTOFFS)
        tai == _TAI_CUTOFFS[i] - Second(1) &&
            throw(ArgumentError("JSOC TAI instant is an unrepresentable UTC leap second"))
    end
    for i in length(_TAI_CUTOFFS):-1:1
        if tai >= _TAI_CUTOFFS[i]
            utc = tai - Second(_TAI_MINUS_UTC[i])
            _UTC_FIRST <= utc < _UTC_STOP_EXCLUSIVE ||
                throw(ArgumentError("UTC instant is outside the verified 2009-01-01 to 2027-06-28 table range"))
            return utc
        end
    end
    throw(ArgumentError("UTC instant precedes the verified table range"))
end

end
