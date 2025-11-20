
# Run the experiments
run-mem0-add:
	python3 run_experiments.py --technique_type mem0 --method add

run-mem0-search:
	python3 run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30

run-mem0-plus-add:
	python3 run_experiments.py --technique_type mem0 --method add --is_graph

run-mem0-plus-search:
	python3 run_experiments.py --technique_type mem0 --method search --is_graph --output_folder results/ --top_k 30

run-rag:
	python3 run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	python3 run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	python3 run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	python3 run_experiments.py --technique_type zep --method add --output_folder results/

run-zep-search:
	python3 run_experiments.py --technique_type zep --method search --output_folder results/

run-openai:
	python3 run_experiments.py --technique_type openai --output_folder results/
