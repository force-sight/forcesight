# python script that takes all folders in 'results/experiments_8_14/" and passes them into the following function:
# python3 -m prediction.live_model --config final_tests_jitter_default_6_6 --index 0 --epoch best --ip 192.168.0.148 --record_video 1 --video_name folder_name --live 0 --folder <folder_name>

import os
import sys
import subprocess

root_folder = "results/experiments_8_14/"
folders = os.listdir(root_folder)

for folder in folders:
    if folder == ".DS_Store":
        continue
    folder = root_folder + folder
    print(folder)
    command = f"python3 -m prediction.live_model --config final_tests_jitter_default_6_6 --index 0 --epoch best --ip 192.168.0.148 --record_video 1 --video_name " + folder + " --live 0 --folder " + folder
    print(command)
    subprocess.call(command, shell=True)
    print("done")
