dataset:
	data prepare -o data/raw/

experiments:
	speech-experiments
	speaker-experiments
	words-experiments

speech-experiments:
	modeling speech-experiment \
		--datadir data/raw/speechvsmusic/ \
		--outdir models/speech/sonnleitner/ \
		--method sonnleitner \
		--seq-len 177
	modeling speech-experiment \
		--datadir data/raw/speechvsmusic/ \
		--outdir models/speech/acrcloud/ \
		--method acrcloud \
		--seq-len 96
	modeling speech-experiment \
		--datadir data/raw/speechvsmusic/ \
		--outdir models/speech/zapr_alg1/ \
		--method zapr_alg1 \
		--seq-len 80
	modeling speech-experiment \
		--datadir data/raw/speechvsmusic/ \
		--outdir models/speech/zapr_alg2/ \
		--method zapr_alg2 \
		--n-bits 8 \
		--seq-len 804

speaker-experiments:
	modeling speaker-experiment \
		--datadir data/raw/librispeech_40speakers/ \
		--outdir models/speaker/sonnleitner/ \
		--method sonnleitner \
		--seq-len 118
	modeling speaker-experiment \
		--datadir data/raw/librispeech_40speakers/ \
		--outdir models/speaker/acrcloud/ \
		--method acrcloud \
		--seq-len 95
	modeling speaker-experiment \
		--datadir data/raw/librispeech_40speakers/ \
		--outdir models/speaker/zapr_alg1/ \
		--method zapr_alg1 \
		--seq-len 80
	modeling speaker-experiment \
		--datadir data/raw/librispeech_40speakers/ \
		--outdir models/speaker/zapr_alg2/ \
		--method zapr_alg2 \
		--n-bits 16 \
		--seq-len 314

words-experiments:
	modeling words-experiment \
		--datadir data/raw/words/ \
		--outdir models/words/sonnleitner/ \
		--method sonnleitner \
		--seq-len 48
	modeling words-experiment \
		--datadir data/raw/words/ \
		--outdir models/words/acrcloud/ \
		--method acrcloud \
		--seq-len 24
	modeling words-experiment \
		--datadir data/raw/words/ \
		--outdir models/words/zapr_alg1/ \
		--method zapr_alg1 \
		--seq-len 20
	modeling words-experiment \
		--datadir data/raw/words/ \
		--outdir models/words/zapr_alg2/ \
		--method zapr_alg2 \
		--n-bits 32 \
		--seq-len 92
