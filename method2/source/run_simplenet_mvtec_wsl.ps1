param(
    [ValidateSet('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper')]
    [string]$Category = 'bottle'
)

$SimpleNetRoot = '/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method2/source/simplenet'
$MVTecRoot = '/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method1/source/patchcore-inspection/data/mvtec'
$RunName = "official_default_$Category"

wsl.exe -e bash -lc "cd '$SimpleNetRoot' && source /home/test/miniforge3/bin/activate patchcore-gpu && python main.py --gpu 0 --seed 0 --log_group simplenet_mvtec --log_project MVTecAD_Results --results_path results --run_name '$RunName' net -b wideresnet50 -le layer2 -le layer3 --pretrain_embed_dimension 1536 --target_embed_dimension 1536 --patchsize 3 --meta_epochs 40 --embedding_size 256 --gan_epochs 4 --noise_std 0.015 --dsc_hidden 1024 --dsc_layers 2 --dsc_margin .5 --pre_proj 1 dataset --batch_size 8 --num_workers 2 --resize 329 --imagesize 288 -d '$Category' mvtec '$MVTecRoot'"
