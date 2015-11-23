# !/bin/bash

#$1: index of first part
#$1: index of  last part
function process_parts {
    cd ~/FDMS/TME3
    for part in `seq $1 $2`
    do
	nohup python f_df_train.py $part > f_df_train_$part.log &
    done
}

process_parts $1 $2
