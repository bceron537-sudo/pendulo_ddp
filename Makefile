GPUS?= 4
DATOS?= datos/pendulo_real.npz
EPOCHS?= 500
BATCH?= 128
K?= 10
LAMBDA_E?= 0.2
HIDDEN?= 128
LAYERS?= 3

install:
	pip install -r requirements.txt

train:
	export NCCL_P2P_DISABLE=1; \
	export NCCL_IB_DISABLE=1; \
	torchrun --nproc_per_node=$(GPUS) -m src.main_ddp train \
		--datos $(DATOS) \
		--epocas $(EPOCHS) \
		--batch_size $(BATCH) \
		--hidden_dim $(HIDDEN) \
		--n_layers $(LAYERS) \
		--num_workers 8 \
		--k_steps $(K) \
		--lambda_energia $(LAMBDA_E)

predict:
	python src/predict.py

clean:
	rm -rf resultados/*.pth resultados/pred

.PHONY: install train predict clean
