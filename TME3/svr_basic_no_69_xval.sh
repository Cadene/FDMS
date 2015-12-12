#! /bin/bash

model="svr"
dataset="basic_no69"
script="${model}_${dataset}.py"

dir="~/FDMS/TME3/"

nbSets=10
nbSplits=5
nbSetsPerSplit=$(($nbSets / $nbSplits))

m=1
nbMachines=10

for c in 1 .1 .01 .001
#for c in .1	 
do
    for i in `seq 0 $(($nbSplits - 1))`
    do	
	log="log/${model}_${dataset}_${c}_${i}.log"

	machine=ppti-14-502-$( [ $m -lt 10 ] && echo 0 )$m
	m=$(( ($m + 1) ))
	if [ $m -gt $nbMachines ]
	then
	    m=1
	fi

	start=$(($nbSetsPerSplit*i+1))
	end=$(($nbSetsPerSplit*(i+1)))
	listTest=`seq $start $end`
	listTest=`echo $listTest`

	if [ $m -eq 1 ]
	then
	    ssh ppti-14-502-11 "~/FDMS/TME3/executeCommande.sh $dir $log python $script $c $listTest"
	else
	    ssh ${machine} "~/FDMS/TME3/executeCommande.sh $dir $log python $script $c $listTest"
	fi
    done
done
