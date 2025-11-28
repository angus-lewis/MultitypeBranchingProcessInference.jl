#!/bin/bash

P_PATH=./seir/p 
R0_PATH=./seir/R0 
S_PATH=./seir/S 
T_PATH=./seir/T

mkdir -p ${P_PATH}
mkdir -p ${R0_PATH}
mkdir -p ${S_PATH}
mkdir -p ${T_PATH}

for p in 0_5 0_75 1_0; do
    cp "density_R_0_config_configs_experiments_configse1i1r_infection_rate=0_3_nsteps=25_cov=[1_0]_observation_probability=${p}_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${P_PATH}/p${p}.pdf"
done

for T in 5 10 15 25; do
    cp "density_R_0_config_configs_experiments_configse1i1r_infection_rate=0_3_nsteps=${T}_cov=[1_0]_observation_probability=0_75_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${T_PATH}/T${T}.pdf"
done

for S in 0_25 1_0 4_0; do
    cp "density_R_0_config_configs_experiments_configse1i1r_infection_rate=0_3_nsteps=25_cov=[${S}]_observation_probability=0_75_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${S_PATH}/s${S}.pdf"
done

for R0 in 0_12 0_2 0_3 0_5; do
    cp "density_R_0_config_configs_experiments_configse1i1r_infection_rate=${R0}_nsteps=25_cov=[1_0]_observation_probability=0_75_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${R0_PATH}/rate${R0}.pdf"
done

P8_PATH=./se8i8r/p 
R08_PATH=./se8i8r/R0 
S8_PATH=./se8i8r/S 
T8_PATH=./se8i8r/T

mkdir -p ${P8_PATH}
mkdir -p ${R08_PATH}
mkdir -p ${S8_PATH}
mkdir -p ${T8_PATH}

for p in 0_5 0_75 1_0; do
    cp "density_R_0_config_configs_experiments_configse8i8r_infection_rate=0_3_nsteps=25_cov=[1_0]_observation_probability=${p}_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${P8_PATH}/p${p}.pdf"
done

for T in 5 10 15 25; do
    cp "density_R_0_config_configs_experiments_configse8i8r_infection_rate=0_3_nsteps=${T}_cov=[1_0]_observation_probability=0_75_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${T8_PATH}/T${T}.pdf"
done

for S in 0_25 1_0 4_0; do
    cp "density_R_0_config_configs_experiments_configse8i8r_infection_rate=0_3_nsteps=25_cov=[${S}]_observation_probability=0_75_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${S8_PATH}/s${S}.pdf"
done

for R0 in 0_12 0_2 0_3 0_5; do
    cp "density_R_0_config_configs_experiments_configse8i8r_infection_rate=${R0}_nsteps=25_cov=[1_0]_observation_probability=0_75_method=hybrid_yaml_Gaussian-Hybrid-Particle.pdf" "${R08_PATH}/rate${R0}.pdf"
done