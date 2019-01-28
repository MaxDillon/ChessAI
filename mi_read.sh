#!/bin/bash

java -cp target/classes:$(cat .classpath) maximum.industries.ReadInstancesKt $*
