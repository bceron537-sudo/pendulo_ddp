import argparse
import os
from src.model_torch_ddp import train_model_ddp, setup, cleanup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="modo train")
    parser.add_argument("--datos", type=str, required=True)
    parser.add_argument("--epocas", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ruta_guardado", type=str, default="./resultados")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--k_steps", type=int, default=10)
    parser.add_argument("--lambda_energia", type=float, default=0.2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(local_rank, world_size)

    train_model_ddp(
        args.datos,
        args.hidden_dim,
        args.n_layers,
        args.epocas,
        args.batch_size,
        args.lr,
        args.ruta_guardado,
        args.num_workers,
        compile=True,
        k_steps=args.k_steps,
        lambda_energia=args.lambda_energia
    )

    cleanup()

if __name__ == "__main__":
    main()