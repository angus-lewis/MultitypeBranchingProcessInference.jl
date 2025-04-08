"""
Estimate day of the week effect from observed data using a simple rolling average

Parameters
    observations: vector of data used to estimate dow effect
    dow_offset: integer offset for the first day of observations, relative to the prior. 
        If dow_offset=n then the first day of observations is day 1+n of the week.
    dow_prior: if a vector is passed, then this is used as prior information to esimate the dow effect,
        if nothing is passed, then no prior is used (need at least 1 week of data in this case). Vector 
        input must have length week_length.
    week_length: the length of the effect (e.g., 7 for weekly effects, 14 days for fortnightly).

Returns
    Vector containing the effect estimates.
"""
function estimate_dow_effect(observations, dow_offset=0, dow_prior=zeros(week_length), week_length=7)
    @assert length(dow_prior) == week_length

    if dow_prior !== nothing
        dow_effect = similar(dow_prior)
        dow_effect .= dow_prior
    else
        dow_effect = zeros(week_length)
    end

    nfullweeks = length(observations)÷week_length
    if dow_prior === nothing 
        @assert nfullweeks > 0 || error("More than one week's worth of observations is required to estimate effect in the absence of prior information.")
    end
    for week_number in 1:nfullweeks
        week_obs = observations[(week_number-1)*week_length .+ (1:week_length)]
        week_mean = mean(week_obs)
        for day in 1:week_length
            dow_idx = (dow_offset + day - 1)%week_length + 1
            dow_effect[dow_idx] += (week_obs[day]/week_mean)
        end
    end

    remaining_days_count = length(observations)%week_length
    remaining_days_idx = nfullweeks*week_length .+ (1:remaining_days_count)
    remaining_days_mean = mean(observations[remaining_days_idx])
    for day in remaining_days_idx
        dow_idx = (dow_offset + day - 1)%week_length + 1
        dow_effect[dow_idx] += (observations[day]/remaining_days_mean)
        # divide by number of full weeks + prior + remaining week
        dow_effect[dow_idx] /= (nfullweeks + (dow_prior!==nothing) + 1)
    end

    for day in nfullweeks*week_length .+ ((remaining_days_count + 1):week_length)
        dow_idx = (dow_offset + day - 1)%week_length + 1
        # divide by number of full weeks + prior
        dow_effect[dow_idx] /= (nfullweeks + (dow_prior!==nothing))
    end

    return dow_effect
end