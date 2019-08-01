#!/bin/bash

sacct --format='JobID%-15,JobName%-15,State,ExitCode,Elapsed,Timelimit' "$@"
