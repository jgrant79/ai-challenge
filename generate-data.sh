#!/bin/sh

VERILOG_DIR=verilog
DATA_DIR=generated

# Usage function, shows a rudimentary running of this script:
usage()
{
cat << EOF

usage: $0 <options>

options:
    -p <#>     Number of permutations to generate for each pattern
    -v <#>     Number of vertices to include in the graph
    -h         Show this message

EOF
}

OPTIONAL_ARGS=
# Grab command line args:
while getopts p:v:h opt
do
   case "$opt" in
      p) OPTIONAL_ARGS="${OPTIONAL_ARGS} -p ${OPTARG}";;
      v) OPTIONAL_ARGS="${OPTIONAL_ARGS} -v ${OPTARG}";;
      h) usage; exit 0;;
      \?) usage; exit 1;;
   esac
done

mkdir -p $DATA_DIR
for f in $(find ${VERILOG_DIR} -type f); do
    filename=$(basename $f)
    stem=$(basename $f .v)
    dirname=$(dirname $f)
    label=$(basename $dirname)

    echo "Generating data for ${filename}..."
    ./src/makegraph.py $f -o ${DATA_DIR}/${stem}.npy -l $label ${OPTIONAL_ARGS}
done
