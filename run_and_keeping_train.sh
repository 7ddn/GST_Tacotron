#!/bin/bash

echo 'Check if training is running'

cd ~/GST_Tacotron
processID=$(ps -ef | grep Model.py | grep -v "grep" | awk "{print $2}")
echo $processID

if [ -z "$processID"]
then
	echo "Training is not running, restarting"
	nohup python3 Model.py >/dev/null
	echo "Training restarted"
else
	echo "Training has already started"
fi
