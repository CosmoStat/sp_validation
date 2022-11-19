#!/usr/bin/bash

fbase_found='found_ID'
fbase_found_wsh='found_ID_wshapes'
fbase_lf='CFIS3500_THELI'
rm -f ${fbase_found}_all.txt
rm -f ${fbase_found_wsh}_all.txt
rm -f ${fbase_found_random}_all.txt
rm -f ${fbase_lf}_all.txt

for patch in P1 P2 P3 P4 P5 P6 P7; do

  # Patch

  ## Total number of tiles
  wc -l $patch/tiles_P?.txt


  # Final catalogue
  #echo "Final catalogue"

  ## Number of final catalogues (.tgz)
  ntgz=`ls -rtl $patch/final*.tgz | wc -l`
  echo "$ntgz final .tgz cats"

  ## Number of final catalogues (.fits)
  nfits=`ls -rtl $patch/output/run_sp_combined/make_catalog_runner/output/final* | wc -l`
  echo "$nfits final .fits cats"

  ## Number of merged final catalogues
  if [ -e $patch/log_merge_final_gal_cat ]; then
    tail -n 1 $patch/log_merge_final_gal_cat
  fi

  ## Number of tile IDs found in merged catalogue
  if [ -e $patch/sp_output/${fbase_found}.txt ]; then
    wc -l $patch/sp_output/${fbase_found}.txt
    wc -l $patch/sp_output/${fbase_found}.txt >> ${fbase_found}_all.txt
  fi

  ## Number of tile IDs found in merged catalogue with shapes
  if [ -e $patch/sp_output/${fbase_found_wsh}.txt ]; then
    wc -l $patch/sp_output/${fbase_found_wsh}.txt
    wc -l $patch/sp_output/${fbase_found_wsh}.txt >> ${fbase_found_wsh}_all.txt
  fi


  # Random catalogue

  ## Number of final catalogues (.tgz)
  ntgz=`ls -rtl $patch/pipeline_flag*.tgz | wc -l`
  echo "$ntgz random .tgz cats"


  ## Number of random catalogues (.fits)
  if [ -d  $patch/output/run_sp_combined_flag ]; then
    nfits=`ls -rtl $patch/output/run_sp_combined_flag/mask_runner/output/pip* | wc -l`
    echo "$nfits random .fits cats"
  fi

  ## Number of merged random catalogues
  if [ -e $patch/log_merge_final_rand_cat ]; then
    tail -n 1 $patch/log_merge_final_rand_cat
  fi

  ### Number of tiles for random catalogue validation
  if [ -e $patch/sp_output_random/${fbase_found}.txt ]; then
    wc -l $patch/sp_output_random/${fbase_found}.txt
    wc -l $patch/sp_output_random/${fbase_found}.txt >> ${fbase_found_random}_all.txt
  fi

  # LensFit tile IDs
  if [ -e ${fbase_lf}_$patch.list ]; then
    wc -l ${fbase_lf}_$patch.list
    wc -l ${fbase_lf}_$patch.list >> ${fbase_lf}_all.txt
  fi
  
  echo

done

echo -n "number of tiles in ${fbase_found}_all.txt = "
summe.pl ${fbase_found}_all.txt 0

echo -n "number of tiles in ${fbase_found_wsh}_all.txt = "
summe.pl ${fbase_found_wsh}_all.txt 0

echo -n "number of tiles in ${fbase_found_random}_all.txt = "
summe.pl ${fbase_found_random}_all.txt 0

if [ -e ${fbase_lf} ]; then
  echo -n "number of tiles in ${fbase_lf}_all.txt = "
  summe.pl ${fbase_lf}_all.txt 0
fi
