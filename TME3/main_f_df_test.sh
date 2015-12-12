# !/bin/bash

# $1 = nb de machine
# $2 = nb de processus par machine
for i in `seq 1 $1`
do
    machine='ppti-14-502-'$([ $i -le 9 ] && echo 0)$i
    seqStart=$(( ( $i - 1 ) * $2 + 1 ))
    seqEnd=$(( $i * $2 ))
    ssh ${machine} "~/FDMS/TME3/part_f_df_test.sh ${seqStart} ${seqEnd}"
done
