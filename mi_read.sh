#!/bin/bash

java -cp target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.ReadInstancesKt $*
