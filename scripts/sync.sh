#!/bin/bash

this_git_dir=`git rev-parse --show-toplevel`
target="/gm2/app/users/labounty/work/g2_analysis/"
server="labounty@gm2gpvm04.fnal.gov"
rsync -avh $this_git_dir/* $server:$target

echo "All done with sync."