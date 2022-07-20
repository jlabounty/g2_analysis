singularity shell \
    -B /home/labounty/github/pioneer/main/:/simulation \
    -B /home/labounty/github/PIONEER-Analysis/:/analysis \
    -B /home/labounty/github/g2_analysis/other_studies/pileup_energy_leakage_geant/geant_sim:/gm2 \
    -B /data/eliza2/g2/users/labounty:/data \
    --containall \
	/data/eliza2/g2/users/labounty/pioneer_g411.sif 