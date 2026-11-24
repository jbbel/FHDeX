#!/bin/bash
# usage: cg_job.sh RUN D INITFILE STEP TAG [extra inputs...]
run=$1; D=$2; init=$3; step=$4; tag=$5; shift 5
cd /data/scratch/jbb/copy/FHDeX/exec/dean_kow/cluster || exit 1
export OMP_NUM_THREADS=1
out=runs/${tag}
[ -s runs/log_${tag}.txt ] && exit 0
exec nice -n 10 ./main3d.gnu.MPI.ex inputs_cg_dk diff_coeff=$D cg.init_file=$init cg.init_step=$step cg.out_dir=$out amr.plot_dt=-1 amr.check_int=-1 cg.write_state=0 "$@" > runs/log_${tag}.txt 2>&1 < /dev/null
