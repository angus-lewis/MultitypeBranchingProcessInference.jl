include("analysis.jl")

const nobs_vec = [10; 15; 25]
const infection_rate_vec = [0.12; 0.3; 0.5]#[0.12; 0.2; 0.3; 0.5]
const obs_prob_vec = [0.5; 0.75; 1.0]
const cov_vec = [0.25; 1.0; 4.0]

# methods - length
for cov in cov_vec[2]
    for nobs in nobs_vec
        for infection_rate in infection_rate_vec[2]
            for obs_prob in obs_prob_vec[2]
                main([
                    "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.0";
                    nobs==nobs_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# method - R0
for cov in cov_vec[2]
    for nobs in nobs_vec[3]
        for infection_rate in infection_rate_vec
            for obs_prob in obs_prob_vec[2]
                main([
                    "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.0";
                    infection_rate==infection_rate_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# method - obs_prob
for cov in cov_vec[2]
    for nobs in nobs_vec[3]
        for infection_rate in infection_rate_vec[2]
            for obs_prob in obs_prob_vec
                main([
                    "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.0";
                    obs_prob==obs_prob_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# method - obs_var
for cov in cov_vec
    for nobs in nobs_vec[3]
        for infection_rate in infection_rate_vec[2]
            for obs_prob in obs_prob_vec[2]
                main([
                    "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.0";
                    cov==cov_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# SENINR
# methods - length
for cov in cov_vec[2]
    for nobs in nobs_vec
        for infection_rate in infection_rate_vec[2]
            for obs_prob in obs_prob_vec[2]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    nobs==nobs_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# method - R0
for cov in cov_vec[2]
    for nobs in nobs_vec[3]
        for infection_rate in infection_rate_vec
            for obs_prob in obs_prob_vec[2]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    infection_rate==infection_rate_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# method - obs_prob
for cov in cov_vec[2]
    for nobs in nobs_vec[3]
        for infection_rate in infection_rate_vec[2]
            for obs_prob in obs_prob_vec
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    obs_prob==obs_prob_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end

# method - obs_var
for cov in cov_vec
    for nobs in nobs_vec[3]
        for infection_rate in infection_rate_vec[2]
            for obs_prob in obs_prob_vec[2]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    cov==cov_vec[1] ? "true" : "false";
                ])
            end
        end
    end
end
