MODEL=MFEA_DaS
BENCHMARK=WCCI22
cp pyMSOO/RunModel/$MODEL/run.py run.py
cp pyMSOO/RunModel/$MODEL/cfg.yaml cfg.yaml

python run.py --nb_generations 10 \
              --name_benchmark $BENCHMARK \
              --number_tasks 50 \
              --ls_id_run '1-1-1' \
              --nb_run 30 \
              --save_path ./RESULTS/$MODEL/$BENCHMARK/\
