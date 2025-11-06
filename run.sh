#!/bin/bash
ssh pi@192.168.17.1 'bash -s' <<'ENDSSH'
~/scripts/launch.sh
~/scripts/run.sh
ENDSSH
