export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7


torchrun --nproc_per_node=7 examples/multiview.py -i assets/multiview/urban_freeway.json -o outputs/multiview_text2world --inference-type=text2world