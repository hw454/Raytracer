#!/bin/bash
# Keith Briggs 2020-11-02
# bash Hayley_shell-script_example_00.sh

for ((job=0; job<25; job++)); do
  python3 Hayley_python_argv_example_00.py "${job}"
done
