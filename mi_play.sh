#!/bin/bash

java -Djava.library.path=/usr/lib/x86_64-linux-gnu -Xmx6g -Xms6g -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.PlayKt $*
