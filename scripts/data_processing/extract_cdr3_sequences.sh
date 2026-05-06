#!/bin/bash

INPUT_DIR="/home/jtepperik/thesis/data/reference_final"
OUTPUT="/home/jtepperik/thesis/data/cdr3_sequences.csv"

echo "pdb,chain,sequence" > "$OUTPUT"

for f in "$INPUT_DIR"/*.pdb; do
  [ -e "$f" ] || continue

  pdb=$(basename "$f" .pdb)

  awk -v pdb="$pdb" '
    BEGIN {
      aa["ALA"]="A"; aa["ARG"]="R"; aa["ASN"]="N"; aa["ASP"]="D"
      aa["CYS"]="C"; aa["GLN"]="Q"; aa["GLU"]="E"; aa["GLY"]="G"
      aa["HIS"]="H"; aa["ILE"]="I"; aa["LEU"]="L"; aa["LYS"]="K"
      aa["MET"]="M"; aa["PHE"]="F"; aa["PRO"]="P"; aa["SER"]="S"
      aa["THR"]="T"; aa["TRP"]="W"; aa["TYR"]="Y"; aa["VAL"]="V"
    }
    substr($0,1,4)=="ATOM" {
      atom  = substr($0,13,4)
      resn  = substr($0,18,3)
      chain = substr($0,22,1)
      resnum = substr($0,23,4) + 0
      icode  = substr($0,27,1)

      # CA only, TCR beta chain (E), CDR3 +-3 residues (IMGT 105-117 -> 102-120)
      if (atom ~ /CA/ && chain == "E" && resnum >= 102 && resnum <= 120) {
        letter = (resn in aa) ? aa[resn] : "X"
        if (!(chain in seen)) {
          order[++nchains] = chain
          seen[chain] = 1
        }
        seq[chain] = seq[chain] letter
      }
    }
    END {
      for (i = 1; i <= nchains; i++) {
        c = order[i]
        if (length(seq[c]) > 0)
          printf "%s,%s,%s\n", pdb, c, seq[c]
      }
    }
  ' "$f" >> "$OUTPUT"

done
