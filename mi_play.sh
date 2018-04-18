#!/bin/bash

java -Xmx6g -Xms6g -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.PlayKt $*
