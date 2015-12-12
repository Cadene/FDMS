#! /bin/bash

model="rfr"
dataset="basic"
script="${model}_${dataset}.py"

dir="~/FDMS/TME3/"

nbSets=10
nbSplits=5
nbSetsPerSplit=$(($nbSets / $nbSplits))

firstM=12
lastM=15
m=$firstM
m=13

#for n in 1000 10000 50000
for n in 10000
do
    for d in 2 3 5
    do
	#for i in `seq 0 $(($nbSplits - 1))`
	for i in `seq 0 3`
		 #a faire: i = 4
	do	
	    log="log/${model}_${dataset}_${n}_${d}_${i}.log"
	    
	    machine=ppti-14-502-$( [ $m -lt 10 ] && echo 0 )$m
	    m=$(( ($m + 1) ))
	    if [ $m -gt $lastM ]
	    then
		m=$firstM
	    fi

	    start=$(($nbSetsPerSplit*i+1))
	    end=$(($nbSetsPerSplit*(i+1)))
	    listTest=`seq $start $end`
	    listTest=`echo $listTest`

	    ssh ${machine} "~/FDMS/TME3/executeCommande.sh $dir $log python $script $n $d $listTest"
	done
    done
done
