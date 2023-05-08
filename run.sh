MODEL=MFEA
BENCHMARK=WCCI22
NUMBER_TASKS=50
cp pyMSOO/RunModel/$MODEL/run.py run.py
cp pyMSOO/RunModel/$MODEL/cfg.yaml cfg.yaml

python run.py --nb_generations 1000 \
              --name_benchmark $BENCHMARK \
              --number_tasks $NUMBER_TASKS \
              --ls_id_run '1' \
              --nb_run 1 \
              --eta 0 \
              --save_path ./RESULTS2/$MODEL/$BENCHMARK/
