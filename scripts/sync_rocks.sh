#!/bin/bash

# "emeraldwalk.runonsave": {
#         "commands":[
#             {
#                 "match": ".*",
#                 "isAsync": true,
#                 "cmd": "bash /home/jlab/github/g2_analysis/scripts/sync.sh > /home/jlab/github/g2_analysis/scripts/sync.log"
#             }
#         ]
# }

this_git_dir="/home/jlab/github/g2_analysis"
target="/home/labounty/github/"
server="labounty@cenpa-rocks.npl.washington.edu"
rsync -avh --exclude '*.root' --exclude '*.csv' --exclude '*.pickle' $this_git_dir $server:$target
exit_status=$?

if [ $exit_status -ne 0 ]; then
    echo "Error"
    powershell.exe -executionpolicy bypass -command New-BurntToastNotification "-Text 'ERROR: Sync to ROCKS not completed!'"   
fi
echo "All done with sync."