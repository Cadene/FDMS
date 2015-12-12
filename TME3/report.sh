#! /bin/bash



for i in `seq 1 15`
do
    machine=ppti-14-502-$( [ $i -lt 10 ] && echo 0 )$i
    if [ $# -ge 1 ]
    then
	echo $machine
	ssh $machine ps u -u 3152691 | grep $1
	#ssh $machine ps u -u 2906480 | grep $1
	echo
    else
	ssh $machine ps u -u 3152691
    fi
done


