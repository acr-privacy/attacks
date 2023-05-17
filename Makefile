.PHONY: prepare-datasets eval-all eval-speech eval-speakers eval-words fingerprints clean clean-data lint

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = acr-privacy-attacks
PYTHON_INTERPRETER = python3
LABELENCODERS_DIR := ${PROJECT_DIR}/src/attacking/labelencoders


#################################################################################
# COMMANDS                                                                      #
#################################################################################

prepare-datasets:
	mkdir ${FINGERPRINT_ROOT}
	tar -xvf acr_privacy_attacks_datasets.tar.gz --directory=${FINGERPRINT_ROOT}

## Run all experiments
eval-all: eval-speech eval-speakers eval-words


## Evaluate speech attacks
eval-speech:
	${PROJECT_DIR}/src/attacking/eval_speech.sh ${PROJECT_DIR} ${EXPERIMENT_ROOT}/music_vs_speech


## Evaluate speaker attacks
eval-speakers: 
	${PROJECT_DIR}/src/attacking/eval_speakers.sh ${PROJECT_DIR} ${EXPERIMENT_ROOT}/speakers


## Evaluate words attacks
eval-words: ${LABELENCODERS_DIR}/speechcommands_words_35_labelencoder.joblib
	${PROJECT_DIR}/src/attacking/eval_words.sh ${PROJECT_DIR} ${EXPERIMENT_ROOT}/words

${LABELENCODERS_DIR}/speechcommands_words_35_labelencoder.joblib:
	$(PYTHON_INTERPRETER) ${LABELENCODERS_DIR}/create_words_labelencoder.py $@


## Generate the fingerprints for all the files in data/processed
fingerprints:
	@cd "src/fingerprints/" && $(PYTHON_INTERPRETER) make_fingerprints.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".mypy_cache" -delete

## Delete downloaded data files
clean-data:
	find data -type f -name "*.wav" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
