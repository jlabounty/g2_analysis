./gridSetupAndSubmitGM2Data.sh --daq --ana \
    --fhicl calo_position_dependence.fcl \
    --sam-dataset gm2pro_daq_offline_dqc_run3D_5218A \
    --njobs 350 \
    --output-dir /pnfs/GM2/scratch/users/labounty/gridoutput \
    --localArea \
    --memory 4 \
    --lifetime 24h \
    # --sam-max-files 2 \
    # --noGrid \