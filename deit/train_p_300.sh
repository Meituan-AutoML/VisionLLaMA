#sh dist_train.sh pllama_wols_base_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/pllama_wols_base_patch16_224_300   --batch 128  --no-repeated-aug   --seed 0 --drop-path 0.3  --dist-eval &> log_txt/pllama_wols_base_patch16_224_300.log

#sh dist_train.sh pllama_wols_large_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/pllama_wols_large_patch16_224_300   --batch 128  --no-repeated-aug   --seed 0 --drop-path 0.5  --dist-eval &> log_txt/pllama_wols_large_patch16_224_300.log

#sh dist_train.sh pllama_wols_small_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/pllama_wols_small_patch16_224_300   --batch 128  --no-repeated-aug   --seed 0 --drop-path 0.2  --dist-eval &> log_txt/pllama_wols_small_patch16_224_300.log

