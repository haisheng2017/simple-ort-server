#!/usr/bin/env bash

SHELL_DIR=$(dirname $(readlink -f "$0"))
cd $SHELL_DIR
git submodule init
git submodule update --remote

cd $SHELL_DIR/lib/web-framework
mvn clean install

MVN_LOCAL=$HOME/.m2/repository