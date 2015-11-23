# !/bin/bash

for i in `seq 1 4`
do
    part=$(($1 * 4 + $i))
    nohup python f_df_train.py $part > f_df_train_$part.log &
done
	 
	 
	 
