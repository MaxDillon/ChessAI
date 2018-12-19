#!/bin/bash

java -cp target/classes:$(cat .classpath) maximum.industries.EvalKt $* \
    | egrep -v "(INFO|DEBUG|ERROR)"
