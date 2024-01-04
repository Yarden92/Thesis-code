echo "=============================== Training model #1: 0.19 ==============================="
python /home/yarcoh/projects/thesis-code4/apps/deep/model_generators/single_model_generator.py --config_path /home/yarcoh/projects/thesis-code4/config/model_generator/skip_unet_channel2_big_c_ds_noiseless_mu019.yml
echo "=============================== Training model #2: 0.38 ==============================="
python /home/yarcoh/projects/thesis-code4/apps/deep/model_generators/single_model_generator.py --config_path /home/yarcoh/projects/thesis-code4/config/model_generator/skip_unet_channel2_big_c_ds_noiseless_mu038.yml
echo "=============================== Training model #3: 0.57 ==============================="
python /home/yarcoh/projects/thesis-code4/apps/deep/model_generators/single_model_generator.py --config_path /home/yarcoh/projects/thesis-code4/config/model_generator/skip_unet_channel2_big_c_ds_noiseless_mu057.yml
echo "=============================== All Done ! ==============================="