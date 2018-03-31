#!/bin/bash

java -cp target/quest-1.0-SNAPSHOT.jar:$(mvn dependency:build-classpath | grep -v INFO) max.dillon.AppKt $*
