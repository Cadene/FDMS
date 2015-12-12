#! /bin/bash

function executeCmd {
    dir=$1
    log=$2
    cmd=${*:3}
    cd $dir
    echo "Starting \"${cmd}\" on $HOSTNAME. Logs in ${log}."
    nohup ${cmd} > ${log} &
}

executeCmd $@
