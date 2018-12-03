#!/bin/bash

touch exit_run_train
kill $*
sleep 5
rm exit_run_train
