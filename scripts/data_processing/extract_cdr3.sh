#!/bin/bash

INPUT_DIR="/home/jtepperik/thesis/data/reference_final"
OUTPUT_DIR="/home/jtepperik/thesis/data/cdr3_structures"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.pdb; do
  [ -e "$f" ] || continue

  base=$(basename "$f" .pdb)

  awk -v base="$base" -v outdir="$OUTPUT_DIR" '
    substr($0,1,4)=="ATOM" {
      chain = substr($0,22,1)
      resnum = substr($0,23,4) + 0
      icode = substr($0,27,1)

      if (resnum >= 105 && resnum <= 117) {
        outfile = outdir "/" base "_chain_" chain "_cdr3.pdb"
        print >> outfile
        files[outfile]=1
      }
    }
    END {
      for (f in files)
        print "END" >> f
    }
  ' "$f"

done