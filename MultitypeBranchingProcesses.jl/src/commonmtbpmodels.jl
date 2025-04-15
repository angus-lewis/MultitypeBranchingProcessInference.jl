# PoissonProcess
function PoissonProcess(rate::AbstractFloat, typeof_state=Int)
    # Poisson process needs two types (states), state 1 remains constant with value 1
    # and has rate λ to model the interarrival times (hence they are iid exponential).
    # State 2 has rate 0 and counts the number of events. The process is always initialised 
    # to state [1, 0].
    initial_dist_cdf = [one(rate)]
    initial_dist_events = [typeof_state[1, 0]]
    initial_dist = MTBPDiscreteDistribution(initial_dist_cdf, initial_dist_events)
    # There is only 1 progeny event, the state which models exponential interarrivals does 
    # not change, (dies and is replaces), and the counter state increments by 1.
    progeny_cdf = [one(rate)]
    progeny_events = [typeof_state[0, 1]]
    progeny_dist = MTBPDiscreteDistribution(progeny_cdf, progeny_events)
    # The counting state needs a valid progeny distribution, even though it 
    # will never occur.
    dummy_progeny_dist = progeny_dist
    # The rate for the interarrival state (1) is non-zero, and there are no events corresponding 
    # to the counting state, so it has rate 0.
    rates = [rate, zero(rate)]
    poisson_process = MultitypeBranchingProcess(2, initial_dist, [progeny_dist, dummy_progeny_dist], rates)
    return poisson_process 
end

# SEIR
"""
Create a MultitypeBranchingProcess instance for a SEIR model
"""
function SEIR(
    N, M, 
    infection_rate::T, 
    exposed_stage_chage_rate::T,
    infectious_stage_chage_rate::T, 
    observation_probablity::T,
    notification_rate::Union{T, Nothing},
    immigration_rates::Union{AbstractArray{T}, Nothing},
    initial_dist::MTBPDiscreteDistribution,
) where {T}
    if immigration_rates !== nothing 
        @assert length(immigration_rates)==N+M "length of immigration rates must match the number exposed and infectious stages N+M"
    end
    # State space has the following interpretation
    # [ E1; ... EN; I1; ... IM;  O; IM]
    # Ei - Exposed stage i
    # Ii - Infectious stage i
    # O  - Observed infectious count
	# N - Observed notification count (this state exists only if its rate !== nothing)
    # IM - Immigtaion (this state exists only if its rates !== nothing)
    # IM state remains constant (i.e., poisson immigration events)
    ntypes = N+M + 1 + (notification_rate!==nothing) + (immigration_rates!==nothing)
    S = variabletype(initial_dist)

    # define all progeny events
    # Define the changes to the state space
	# Infection
    infection_event = zeros(S, ntypes)
    infection_event[1] = one(S)

	# Progression through latent stages
    stage_progression_events = Array{S,1}[]
    for state_idx in 1:(N+M-1)
        event = zeros(S, ntypes)
        event[state_idx] = -one(S)
        event[state_idx+1] = one(S)
        push!(stage_progression_events, event)
    end

    recovery_event = zeros(S, ntypes)
    recovery_event[N+M] = -one(S)

	# Event observation occurs simulataneously with EN -> I1, with specified observation probability
    observation_event = copy(stage_progression_events[N])
    observation_event[N+M+1] = one(S)

    # Notification after delay
    if notification_rate !== nothing
	    notification_event = zeros(S, ntypes)
	    notification_event[N+M+1] = -one(S)
	    notification_event[N+M+2] = one(S)
    end

    if immigration_rates !== nothing
        immigration_events = Array{S,1}[]
        for state_idx in 1:(N+M)
            event = zeros(S, ntypes)
            event[state_idx] = one(S)
            push!(immigration_events, event)
        end
    end

    # define progeny distributions themselves
    progeny_dist_type = MTBPDiscreteDistribution{
        Vector{T}, # CDF parameters
        Vector{Vector{S}}, # Events
        Vector{T}, # first moments
        Matrix{T}, # second moments
    }
    progeny_dists = progeny_dist_type[]

    # unobserved exposed states
    unobserved_exposed_state_cdf = T[1]
    for state_idx in 1:N-1
        # only stage transitions occur while exposed
        unobserved_exposed_state_events = [
            stage_progression_events[state_idx]
        ]
        dist = MTBPDiscreteDistribution(
            unobserved_exposed_state_cdf, unobserved_exposed_state_events
        )
        push!(progeny_dists, dist)
    end

    # observed exposed state
    observed_exposed_state_cdf = T[(1-observation_probablity), 1]
    observed_exposed_state_events = [
        stage_progression_events[N],
        observation_event,
    ]
    dist = MTBPDiscreteDistribution(observed_exposed_state_cdf, observed_exposed_state_events)
    push!(progeny_dists, dist)

    # infectious state transitions without recovery
    infectious_state_cdf = T[infection_rate/(infection_rate+infectious_stage_chage_rate),1] 
    for state_idx in N+1:N+M-1
        # stage transitions or infecitons can occur
        infectious_state_events = [
            infection_event,
            stage_progression_events[state_idx],
        ]
        dist = MTBPDiscreteDistribution(infectious_state_cdf, infectious_state_events)
        push!(progeny_dists, dist)
    end

    # infectious state transitions with recovery 
    infectious_state_cdf = T[infection_rate/(infection_rate+infectious_stage_chage_rate),1]
    infectious_state_events = [infection_event, recovery_event]
    dist = MTBPDiscreteDistribution(infectious_state_cdf, infectious_state_events)
    push!(progeny_dists, dist)

    # Observation state
	# observation_state_cdf = T[1]
    if notification_rate !== nothing
        observation_state_events = [notification_event]
        dist = MTBPDiscreteDistribution(T[1], observation_state_events)
        push!(progeny_dists, dist)
    else
        dist = dummy_mtbp_discrete_distribution(ntypes, typeof(infection_event), T)
        push!(progeny_dists, dist)
    end

    # Notification state - persist throughout simulation
    if notification_rate !== nothing
        dist = dummy_mtbp_discrete_distribution(ntypes, typeof(infection_event), T)
        push!(progeny_dists, dist)
    end

    if immigration_rates !== nothing
        total_immigration_rate = sum(immigration_rates)
        if total_immigration_rate==zero(total_immigration_rate)
            immigration_cdf = collect(
                Iterators.drop(
                    range(
                        zero(total_immigration_rate),
                        one(total_immigration_rate),
                        length(immigration_rates) + 1,
                    ),
                    1,
                ),
            )
        else
            immigration_pmf = immigration_rates./total_immigration_rate
            immigration_cdf = cumsum(immigration_pmf)
            # ensure cdf is proper
            immigration_cdf ./= immigration_cdf[end]
        end
        immigration_progeny_dist = MTBPDiscreteDistribution(immigration_cdf, immigration_events)
        push!(progeny_dists, immigration_progeny_dist)
    end

    rates = zeros(T, ntypes)
	rates[1:N] .= exposed_stage_chage_rate
	rates[N+1:N+M] .= infection_rate + infectious_stage_chage_rate
    if notification_rate !== nothing
	    rates[N+M+1] = notification_rate
        if immigration_rates !== nothing
            rates[N+M+3] = total_immigration_rate
        end
    else
        if immigration_rates !== nothing
            rates[N+M+2] = total_immigration_rate
        end
    end

    return MultitypeBranchingProcess(ntypes, initial_dist, progeny_dists, rates)
end

"""
SEIR taking in the initial state as a vector
"""
function SEIR(
    N, M, 
    infection_rate::T, 
    exposed_stage_chage_rate::T, 
    infectious_stage_chage_rate::T, 
    observation_probablity::T,
    notification_rate::Union{T, Nothing}=nothing,
    immigration_rates::Union{AbstractArray{T}, Nothing}=nothing,
    initial_state = _default_initial_state(N, M, notification_rate, immigration_rates),
) where {T}
    initial_cdf = T[1]
    initial_dist = MTBPDiscreteDistribution(initial_cdf, [initial_state])
    return SEIR(
        N, M, 
        infection_rate, 
        exposed_stage_chage_rate, 
        infectious_stage_chage_rate, 
        observation_probablity, 
        notification_rate,
        immigration_rates, 
        initial_dist
    )
end

function _default_initial_state(N, M, notification_rate, immigration_rate)
    return [1; zeros(Int, N + M); 0; 1]
end
function _default_initial_state(N, M, notification_rate::Nothing, immigration_rate)
    return [1; zeros(Int, N + M); 1]
end
function _default_initial_state(N, M, notification_rate, immigration_rate::Nothing)
    return [1; zeros(Int, N + M); 0]
end
function _default_initial_state(N, M, notification_rate::Nothing, immigration_rate::Nothing)
    return [1; zeros(Int, N + M)]
end