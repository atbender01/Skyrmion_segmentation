"""
Created: A. Bender, ANL, CMU

Python script for generating and running mumax scripts
can be run on single gpu, or on a subset of the available gpus
for the multi-gpu case, can/should be improved so that it's not waiting for all jobs
to clear before adding more (see notes below)
"""

import numpy as np
import os as os
from pathlib import Path
import sys as sys
import datetime
import textwrap
import subprocess
import time
import random
import math

# basic setup

single_test = False
check_num = 5
make_pngs = True
stime = time.time()
region_map = True # region_map will vary the Msat by region
with_and_without_regions = True
with_and_without_regions_ratio = [9, 1] # ratio between mx3s (with, without)
iterations = 10000

# names
path_name = "/zota/Lorentz/AlecBender/mumax_training_files"
output_name = "12"


gpus = [0, 1, 2]
print(f"Running with GPUs {gpus}\n")
output_format = "OVF1_TEXT"

# defining geometry
PBC = (2, 2, 0)  # x, y, z
grid_size = (256, 256, 1)  # x, y, z ### will set z-scale based on thickness
cell_size = (3e-9, 3e-9, 27.5e-9)  # x, y, z
tvalues = [27.5e-9, 27.5e-9]  # thickness values to use (minval, maxval)
# too large of cell sizes and mumax doesnt work very well

# Magnetization Parameter Ranges - loosely FGT skyrmions
# (minval, maxval)
# Msats safe range = [1.00e+05, 4.00e+05]
Msats = [1.50e+05, 3.00e+05]  # A/m, 1.1e5, larger means fewer mazes
# Aexs safe range = [1.1e-11, 5.1e-11]
Aexs = [1.1e-11, 1.9e-11]  # J/m 1e-11
Ku1s = [1.50e+06]  # J/m^3 1.5e6
# Dinds safe range = [8.0e-04, 1.40e-03]
Dinds = [1.00e-03, 1.30e-03]  # 1e-3 # larger means more maze domains
alphas = [0.25]  # 0.25
# Bexts safe range = [0.30, 0.50]
Bexts = [0.40, 0.40]  # T 0.49
region_amount = 256
grain_size = 300e-9
run_num = 0

# Folder creation 

fnames = []
wd = Path(f"{path_name}")
dataset_name = f"SkyrmTrain_{output_name}"
savedir = wd / f"SkyrmNet_training{output_name}"
savedir.mkdir(exist_ok=True)
magdir = savedir / "magnetizations"
magdir.mkdir(exist_ok=True)
imdir = savedir / "training_images"
imdir.mkdir(exist_ok=True)
labeldir = savedir / "label_images"
labeldir.mkdir(exist_ok=True)
mx3dir = savedir / "mx3s"
mx3dir.mkdir(exist_ok=True)

# region_sim sets the Aex and Dind
# also returns a list of the Msat values for each region
def region_sim (iterations):

    # setting the min and max values
    Aex_max = Aexs[1]
    Aex_min = Aexs[0]
    Msat_max = Msats[1]
    Msat_min = Msats[0]
    Dind_max = Dinds[1]
    Dind_min = Dinds[0]

    region_sim_list = []

    for _ in range(iterations):
        seed = random.randint(0, 1e9)
        Aex = random.uniform(Aex_min, Aex_max)
        Dind = random.uniform(Dind_min, Dind_max)
        Msat_set = random.uniform(Msat_min, Msat_max)
        
        set_region = []
        max_Msat = None
        min_Msat = None

        # sets the region Msat as +/- 25% of a random given Msat
        for _ in range(region_amount - 1):
            region_Msat = Msat_set + Msat_set * random.randint(-25, 25) * 0.01
            while (region_Msat < Msat_min) or (region_Msat > Msat_max):
                region_Msat = Msat_set + Msat_set * random.randint(-25, 25) * 0.01

            set_region.append(region_Msat)

            # sets the min and max Msat
            if max_Msat is None and min_Msat is None:
                max_Msat = region_Msat
                min_Msat = region_Msat
            else:
                max_Msat = max(max_Msat, region_Msat)
                min_Msat = min(min_Msat, region_Msat)
    
        avg_Msat = sum(set_region) / len(set_region)
        region_sim_list.append([seed, Aex, avg_Msat, max_Msat, min_Msat, Dind, set_region])

    return region_sim_list

# Creates mx3 files with regioned skyrmions
def with_regions (iterations):
    global run_num
    if single_test:
        iterations = 1
    region_sim_list = region_sim(Aexs, Dinds, iterations)
    for RegionSim in region_sim_list:
        seed, Aex, avg_Msat, max_Msat, min_Msat, Dind, set_region = RegionSim
    
        # set constant parameters
        thickness = tvalues[0]
        alpha = alphas[0]
        Ku1 = Ku1s[0]
        Bext = Bexts[0]

        gridZ = round(thickness / cell_size[2])
        thick_val = int(gridZ * cell_size[2] * 1e9)

        RunNumber = f"{run_num:05}"
        mx3_fname = f"{dataset_name}_{run_num:05}_A{Aex:.2e}_AvgMs{avg_Msat:.2e}_Dind{Dind:.2e}_MaxMs{max_Msat:.2e}_MinMs{min_Msat:.2e}_t{thick_val}_Ku{Ku1:.2e}_B{Bext:.2f}_seed{seed:.2f}_Regions"
        fnames.append(mx3_fname)

        # creates a string to set all region Msat values
        set_all_regions = ""
        for i in range(region_amount - 1):
            region_Msat = set_region[i]
            set_all_regions += f"Msat.SetRegion({i}, {region_Msat})\n"

        d = datetime.datetime.now()
        f = open(mx3dir / (mx3_fname + ".mx3"), "w")
        f.write(
            textwrap.dedent(
                f"""\
// Mumax runtime file for SkyrmNet training
// Created: A. Bender, ANL, CMU {d:%Y-%m-%d %H:%M:%S}

// outputformat = {output_format}
SetPBC({PBC[0]}, {PBC[1]}, {PBC[2]})
setgridsize({grid_size[0]}, {grid_size[1]}, {gridZ})
setcellsize({cell_size[0]}, {cell_size[1]}, {cell_size[2]})

Aex = {Aex:.2e} // J/m
Ku1 = {Ku1:.2e} // J/m^3
avgMsat := {avg_Msat:.2e} // J/m^2
maxMsat := {max_Msat:.2e} // J/m^2
minMsat := {min_Msat:.2e} // J/m^2
Dind ={Dind:.2e}
alpha = {alpha:.2f}
B := {Bext:.2f} // T
seed := {seed:d}
randSeed(seed)
m = randommagseed(seed)

setgeom(rect(256, 256))
grainSize  := {grain_size}  // m

maxRegion  := {region_amount:d} - 1
randomSeed := randInt(1000000000)
ext_makegrains(grainSize, maxRegion, randomSeed)

defregion({region_amount:d}, rect(256, 256).inverse()) // region 256 is outside, not really needed

{set_all_regions}

for i:=0; i<maxRegion; i++{{
    for j:=i+1; j<maxRegion; j++{{
                ext_ScaleExchange(i, j, 0.9)
    }}
}}

B_ext = vector(0, 0, B)

relax()
print(maxAngle*180/3.141592653)
// saveas(regions, \"{RunNumber+".ovf"}\")
saveas(m_full, \"{mx3_fname+".ovf"}\")
"""
            )
        )
        f.close()
        run_num += 1

        if single_test:
            break

# Creates mx3 files with uniform skyrmions
def without_regions (iterations):
    global run_num
    Aex_max = Aexs[1]
    Aex_min = Aexs[0]
    Msat_max = Msats[1]
    Msat_min = Msats[0]
    Dind_max = Dinds[1]
    Dind_min = Dinds[0]
    for x in range(iterations):
        seed = random.randint(0, 1e9)

        Aex = random.uniform(Aex_min, Aex_max)
        Dind = random.uniform(Dind_min, Dind_max)
        Msat = random.uniform(Msat_min, Msat_max)

        alpha = alphas[0]
        thickness = tvalues[0]
        alpha = alphas[0]
        Ku1 = Ku1s[0]
        Bext = Bexts[0]

        gridZ = round(thickness / cell_size[2])
        thick_val = int(gridZ * cell_size[2] * 1e9)
        mx3_fname = f"{dataset_name}_{run_num:05}_A{Aex:.2e}_Ms{Msat:.2e}_Dind{Dind:.2e}_t{thick_val}_Ku{Ku1:.2e}_B{Bext:.2f}_Uniform"
        fnames.append(mx3_fname)

        d = datetime.datetime.now()
        f = open(mx3dir / (mx3_fname + ".mx3"), "w")
        f.write(
            textwrap.dedent(
                f"""\
// Mumax runtime file for SkyrmNet training
// Created: A. Bender, ANL, CMU {d:%Y-%m-%d %H:%M:%S}

outputformat = {output_format}
SetPBC({PBC[0]}, {PBC[1]}, {PBC[2]})
setgridsize({grid_size[0]}, {grid_size[1]}, {1})
setcellsize({cell_size[0]}, {cell_size[1]}, {cell_size[2]})

Msat = {Msat:.2e} // A/m
Aex = {Aex:.2e} // J/m
Ku1 = {Ku1:.2e} // J/m^3
Dind = {Dind:.2e} // J/m^2
alpha = {alpha:.2f}
B := {Bext:.2f} // T

seed := {seed:d}
randSeed(seed)
m = randommagseed(seed)

B_ext = vector(0, 0, B)

relax()
print(maxAngle*180/3.141592653)
saveas(m_full, \"{mx3_fname+".ovf"}\")
"""
            )
        )
        f.close()
        run_num += 1

        if single_test:
            break

# This is where the actual commands to create all the files are taking place
if with_and_without_regions:

    # check to ensure the ratio results in an integer
    region_ratio = with_and_without_regions_ratio[0]
    not_region_ratio = with_and_without_regions_ratio[1]
    ratio_total = region_ratio + not_region_ratio
    assert (iterations % ratio_total) == 0, "Ratio must result in an integer"

    ratio_GCD = math.gcd(region_ratio, not_region_ratio)
    lowest_region_ratio = region_ratio // ratio_GCD
    lowest_not_region_ratio = not_region_ratio // ratio_GCD
    lowest_ratio_total = lowest_region_ratio + lowest_not_region_ratio

    region_iterations = (lowest_region_ratio // lowest_ratio_total) * iterations
    not_region_iterations = (lowest_not_region_ratio // lowest_ratio_total) * iterations

    not_region_number_list = []
    while run_num < iterations:
        # chooses random number(s) to create uniform skyrmions
        # amount of random number(s) chosen is based on the with_and_without_regions_ratio
        for i in range(lowest_not_region_ratio):
            not_region_number_list.append(random.randint(0, (lowest_ratio_total - 1)))

        for j in range(lowest_ratio_total):
            if j not in not_region_number_list:
                with_regions(1)
            else:
                without_regions(1)
        not_region_number_list = []
elif region_map:
    with_regions(iterations)
else:
    without_regions(iterations)

# Now the runtime file is created so we run the mumax program at commandline

def run_sims_single(file_names, gpus, single_test, make_pngs):  # on just one gpu
    cur = 1
    tot = len(file_names)
    gpuint = gpus[0]
    for fname in file_names:
        print(f"\n\nRunning simulation for: {fname}\non gpu {gpuint}")
        print(f"{cur} / {tot}\n\n")
        cur += 1
        # os.system(f"mumax3 {savedir / (fname+'.mx3')}")
        os.system(f"mumax3 -gpu={gpuint} -o={magdir} {mx3dir / (fname+'.mx3')}")

        if single_test or make_pngs:
            os.system(f"mumax3-convert -png -normalize -arrows 10 {magdir}/*.ovf")

        if single_test:
            if cur > check_num:
                print("\n\nBreaking for single test.\n")
                return


def run_sims_multiprocess(file_names, gpus, make_pngs, batchsize=1, overwrite=False):
    """Run mumax sims on multiple GPUs

    Args:
        file_names (list): list of filepaths.mx3 to run
        gpus (list): list of GPU ints to use (e.g. [0,1,2])
        make_pngs (bool): whether or not to make .png outputs of the files, useful when
                          testing.
        batchsize (int, optional): Number of simulations to run on each GPU.
    """
    cur = 1
    ngpus = len(gpus)
    # TODO improve this so isn't waiting unecessarily
    # could make a stack of all the filenames
    # while loop checking number running on each GPU every second or so
    # when number is less than batchsize, adds another one to that gpu
    # until the stack is empty
    # have the same p.wait() at the end

    fnames_temp = []
    if not overwrite:
        for f in file_names:
            ovf = magdir / (f + ".ovf")
            if not ovf.exists():
                fnames_temp.append(f)
    file_names = fnames_temp
    file_names.sort()
    file_names = file_names[::-1]

    tot = len(file_names)

    procs_gpu = []
    cur = 1

    # start the initial jobs
    for _ in range(batchsize):  
        for gpuint in gpus:
            print(f"\n------ Starting new job. {cur}/{tot} ------\n")
            fname = file_names.pop()
            proc = subprocess.Popen(
                [
                    "mumax3",
                    f"-gpu={gpuint}",
                    f"-o={magdir}",
                    f"{mx3dir / (fname+'.mx3')}",
                ]
            )
            procs_gpu.append([proc, gpuint])
            cur += 1
            if len(file_names) == 0:
                break
        if len(file_names) == 0:
            break

    # add new jobs as previous ones complete
    while len(file_names) > 0:
        for p, gpuint in procs_gpu:
            # subprocess is done
            if p.poll() is not None:
                print(f"\n------ Starting new job. {cur}/{tot} ------\n")
                procs_gpu.remove([p, gpuint])
                # add new job
                fname = file_names.pop()
                proc = subprocess.Popen(
                    [
                        "mumax3",
                        f"-gpu={gpuint}",
                        f"-o={magdir}",
                        f"{mx3dir / (fname+'.mx3')}",
                    ]
                )
                procs_gpu.append([proc, gpuint])
                cur += 1
            # subprocess is not done
            else:
                pass
        time.sleep(1)

    # wait for the final runs to complete
    for p, gpuint in procs_gpu:
        p.wait()

    if single_test or make_pngs:
        os.system(f"mumax3-convert -png -normalize -arrows 10 {magdir}/*.ovf")
        if region_map:
            os.system(f"mumax3-convert -png {magdir}/*.ovf")
    return


### select if single or multiple GPUs used

# run_sims_single(fnames, gpus, single_test, make_pngs)
run_sims_multiprocess(fnames, gpus, make_pngs, batchsize=1, overwrite=False)
# batchsize 1 -> ovf file done every 2-3 minutes
# about the same for batchsize of 10.. batchsize doesnt seem to make much difference


ttime = time.time() - stime
days = ttime // (3600 * 24)
hr = ttime // 3600
min = (ttime % 3600) // 60
sec = (ttime % 3600) % 60
print(f"\n\nElapsed time (d:h:m:s): ({days:02.0f}:{hr:02.0f}:{min:02.0f}:{sec:02.0f})")
print("\nFinished\n===========================================")
