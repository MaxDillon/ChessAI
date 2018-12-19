#!/bin/bash

mvn dependency:build-classpath | grep -v INFO | grep -v WARNING | grep -v Downloading > .classpath
