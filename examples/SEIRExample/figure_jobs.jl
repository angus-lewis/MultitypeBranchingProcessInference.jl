include("analysis.jl")

# # methods - length
# for cov in [1.0]
#     for nobs in [30; 60; 90]
#         for infection_rate in [0.04]
#             for obs_prob in [0.75]
#                 main([
#                     "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
#                     "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
#                     "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
#                     "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
#                     "2.0";
#                     nobs==30 ? "true" : "false";
#                 ])
#             end
#         end
#     end
# end

# # method - R0
# for cov in [1.0]
#     for nobs in [90]
#         for infection_rate in [0.024; 0.04; 0.06]
#             for obs_prob in [0.75]
#                 main([
#                     "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
#                     "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
#                     "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
#                     "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
#                     "2.0";
#                     infection_rate==0.024 ? "true" : "false";
#                 ])
#             end
#         end
#     end
# end

# # method - obs_prob
# for cov in [1.0]
#     for nobs in [90]
#         for infection_rate in [0.04]
#             for obs_prob in [0.5; 0.75; 1.0]
#                 main([
#                     "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
#                     "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
#                     "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
#                     "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
#                     "2.0";
#                     obs_prob==0.5 ? "true" : "false";
#                 ])
#             end
#         end
#     end
# end

# # method - obs_var
# for cov in [0.25; 1.0; 4.0]
#     for nobs in [90]
#         for infection_rate in [0.04]
#             for obs_prob in [0.75]
#                 main([
#                     "configs/experiments/configse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
#                     "Gaussian=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
#                     "Hybrid=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
#                     "Particle=data/samplesse1i1r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
#                     "2.0";
#                     cov==0.25 ? "true" : "false";
#                 ])
#             end
#         end
#     end
# end


# SENINR
# methods - length
for cov in [1.0]
    for nobs in [30; 60; 90]
        for infection_rate in [0.04]
            for obs_prob in [0.75]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    nobs==30 ? "true" : "false";
                ])
            end
        end
    end
end

# method - R0
for cov in [1.0]
    for nobs in [90]
        for infection_rate in [0.024; 0.04; 0.06]
            for obs_prob in [0.75]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    infection_rate==0.024 ? "true" : "false";
                ])
            end
        end
    end
end

# method - obs_prob
for cov in [1.0]
    for nobs in [90]
        for infection_rate in [0.04]
            for obs_prob in [0.5; 0.75; 1.0]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    obs_prob==0.5 ? "true" : "false";
                ])
            end
        end
    end
end

# method - obs_var
for cov in [0.25; 1.0; 4.0]
    for nobs in [90]
        for infection_rate in [0.04]
            for obs_prob in [0.75]
                main([
                    "configs/experiments/configse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.yaml";
                    "Gaussian=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=kalman_filter.f64_array.bin";
                    "Hybrid=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=hybrid.f64_array.bin";
                    "Particle=data/samplesse8i8r_infection_rate=$(infection_rate)_nsteps=$(nobs)_cov=[$(cov)]_observation_probability=$(obs_prob)_method=particle_filter.f64_array.bin";
                    "2.4";
                    cov==0.25 ? "true" : "false";
                ])
            end
        end
    end
end
