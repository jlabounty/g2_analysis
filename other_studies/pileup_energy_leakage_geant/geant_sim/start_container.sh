singularity shell \
    -B /home/jlab/github/pioneer/main/:/simulation \
    -B /home/jlab/github/PIONEER-Analysis/:/analysis \
    -B /home/jlab/github/g2_analysis/other_studies/pileup_energy_leakage_geant/geant_sim:/gm2 \
    --containall \
	/home/jlab/github/pioneer/main/MonteCarlo/utils/submission_scripts/pioneer_latest.sif