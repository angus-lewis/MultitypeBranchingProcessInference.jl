function estimate_dow_effect(observations, dowoffset=0, dow_prior=zeros(week_length), week_length=7)
    dow_effect = similar(dow_prior)
    dow_effect .= dow_prior

    nfullweeks = length(observations)÷week_length
    for week_number in 1:nfullweeks
        week_obs = observations[(week_number-1)*week_length .+ (1:week_length)]
        week_mean = mean(week_obs)
        for day in 1:week_length
            dow_idx = (dowoffset + day - 1)%week_length + 1
            dow_effect[dow_idx] += (week_obs[day]/week_mean)
        end
    end

    remaining_days_count = length(observations)%week_length
    remaining_days_idx = nfullweeks*week_length .+ (1:remaining_days_count)
    remaining_days_mean = mean(observations[remaining_days_idx])
    for day in remaining_days_idx
        dow_idx = (dowoffset + day - 1)%week_length + 1
        dow_effect[dow_idx] += (observations[day]/remaining_days_mean)
        # divide by number of full weeks + prior + remaining week
        dow_effect[dow_idx] /= (nfullweeks+2)
    end

    for day in nfullweeks*week_length .+ ((remaining_days_count + 1):week_length)
        dow_idx = (dowoffset + day - 1)%week_length + 1
        # divide by number of full weeks + prior
        dow_effect[dow_idx] /= (nfullweeks+1)
    end

    return dow_effect
end