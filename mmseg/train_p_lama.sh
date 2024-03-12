bash ./tools/dist_train.sh  configs/twins/twins_svt_lama_as-l_uperhead_8xb2-160k_ade20k-512x512.py 8 --amp &> twins_svt_lama_as-l_uperhead_8xb2-160k_ade20k-512x512.log

bash ./tools/dist_train.sh  configs/twins/twins_svt_lama_as-b_uperhead_8xb2-160k_ade20k-512x512.py 8 --amp &> twins_svt_lama_as-b_uperhead_8xb2-160k_ade20k-512x512.log

