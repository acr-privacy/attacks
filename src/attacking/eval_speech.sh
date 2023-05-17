#!/usr/bin/env bash
PROJECT_DIR=$1
RESULTS_DIR=$2

#create results dir if necessary
if [ ! -d "${RESULTS_DIR}" ]; then
	mkdir -p ${RESULTS_DIR}
fi

ALGORITHMS=(SONNLEITNER ACRCLOUD ZAPR_ALG1 ZAPR_ALG2)
for ALGORITHM in ${ALGORITHMS[@]}; do
	OUTDIR=${RESULTS_DIR}/${ALGORITHM}
	mkdir -p ${OUTDIR}
	tar -zxvf ${PROJECT_DIR}/models/speech_vs_music/${ALGORITHM}.tar.gz -C ${OUTDIR}
	python ${PROJECT_DIR}/src/attacking/evaluation/evaluate_speech_vs_music.py -mt ${ALGORITHM} -out ${OUTDIR}
done
