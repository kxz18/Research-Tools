#!/bin/zsh
##########################################################################
# File Name: kill_distributed.sh
# Author: kxz
# mail: 15068701650@163.com
# Created Time: Saturday, December 11, 2021 PM09:19:14 HKT
#########################################################################

if [ -z $1 ]; then
    echo "Usage: bash kill_distributed.sh <keyword>"
    exit 1
else
    ps -ef | grep $1 | grep -v grep | cut -c 9-15 | xargs kill -9
fi