# Requires 24GB of graphics memory
python styless_attack.py --mi --ti --di --si --styless \
    --model wide_resnet101_2 \
    --styNum 8 \
    --save_in

python eval.py --save_dir exp/test_samples/wide_resnet101_2/ifgsm_mi_ti_di_si_styless_8
