#! /bin/bash

model="svr"
dataset="basic_no100_full"
script="test_${model}_${dataset}.py"

dir="~/FDMS/TME3/"

nbSets=10
nbSplits=5
nbSetsPerSplit=$(($nbSets / $nbSplits))

m=1
nbMachines=10

for i in `seq 1 16`
do	
    log="log/test_${model}_${dataset}.log"

    machine=ppti-14-502-$( [ $m -lt 10 ] && echo 0 )$m
    m=$(( ($m + 1) ))
    if [ $m -gt $nbMachines ]
    then
	m=1
    fi

    if [ $m -eq 1 ]
    then
	machine=ppti-14-502-08
    fi    
    ssh ${machine} "~/FDMS/TME3/executeCommande.sh $dir $log python $script $i"
done

