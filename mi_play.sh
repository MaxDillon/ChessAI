#!/bin/bash

java -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.PlayKt $*
