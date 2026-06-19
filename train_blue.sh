#!/usr/bin/env bash
# sh main/tools/gen_scp_all.sh /workdir/denoiser/datasets/ner-humannoise /workdir/denoiser/main
# python main/nnet/trainnew.py --gpus 0, 1 --epochs 50 --checkpoint /workdir/denoiser/MS_R1_SL1_humanspeech__saved_models --batch-size 4 --num-workers 0


# sh main/tools/gen_scp_all.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-human300 /media/speech70809/Data01/speech_donoiser_new/main
#python main/nnet/trainnew.py --gpus 0 --epochs 50 --checkpoint /media/speech70809/Data02/MS_R1_SL2_human3000121__saved_models --batch-size 4 --num-workers 0

# sh main/tools/gen_scp_all.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-human300 /media/speech70809/Data01/speech_donoiser_new/main
# python main/nnet/trainnew.py --gpus 0 --epochs 50 --checkpoint /media/speech70809/Data02/MS_R1_SL2_humannoise300hr__saved_models --batch-size 4 --num-workers 0

#sh main/tools/gen_scp_all.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-braodnoise300hr /media/speech70809/Data01/speech_donoiser_new/main
# python main/nnet/trainnew.py --gpus 0 --epochs 50 --checkpoint /media/speech70809/Data02/MS_R1_SL2_braodnoise300hr__saved_models --batch-size 4 --num-workers 0


#sh main/tools/gen_scp_all.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr /media/speech70809/Data01/speech_donoiser_new/main
# python main/nnet/trainnew_blue.py --gpus 0, 1  --epochs 50 --checkpoint /workdir/denoiser/MS_SL1_1DBPF_model  --batch-size 16 --num-workers 0
# python main/nnet/trainnew.py --gpus 0 --epochs 50 --checkpoint /media/denoiser/Toshibha-3.0TB/convTasnet --batch-size 4 --num-workers 0
# python main/nnet/trainnew_blue.py --gpus 0 --epochs 10 --checkpoint /workdir/denoiser/MS_SL1_1DBPF_model_1 --batch-size 4 --num-workers 0
python trainnew_blue.py --gpus 0 --epochs 50 --checkpoint //media/kaldi/SP1/MingHshuan/checkpoint/20241121_1_test --batch-size 2 --num-workers 0 --trainer_type repeat
# python main/nnet/trainnew_blue.py --gpus 0 --epochs 50 --checkpoint /workdir/denoiser/MS_SL1_1DBPF_model_new1  --batch-size 16 --num-workers 0