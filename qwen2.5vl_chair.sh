

python generate_response_chair.py \
    --device "cuda:0" \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --output "result/chair/Qwen2.5-VL-3B-Instruct/output_greedy.json" \
    --method "greedy" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:0" \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --output "result/chair/Qwen2.5-VL-3B-Instruct/output_beam.json" \
    --method "beam" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:0" \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --output "result/chair/Qwen2.5-VL-3B-Instruct/output_dola.json" \
    --method "dola" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:0" \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --output "result/chair/Qwen2.5-VL-3B-Instruct/output_vcd.json" \
    --method "vcd" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:0" \
    --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --output "result/chair/Qwen2.5-VL-3B-Instruct/output_deco.json" \
    --method "deco" \
    --max_tokens 64 \