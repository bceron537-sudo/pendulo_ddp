import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.dynamics_torch import energia_pendulo_doble

torch.set_float32_matmul_precision('high')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, hidden_dim=128, n_layers=3):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PenduloDataset(Dataset):
    def __init__(self, ruta_npz: str, normalizar: bool = True, k_steps: int = 1):
        data = np.load(ruta_npz)
        estados = data['estados'].astype(np.float32)
        if estados.shape[1] != 4:
            raise ValueError(f"Esperaba 4 columnas, got {estados.shape[1]}")
        self.media = estados.mean(axis=0)
        self.std = estados.std(axis=0) + 1e-8
        self.dt = float(data.get('dt', 0.01))
        self.k_steps = k_steps
        if normalizar:
            estados = (estados - self.media) / self.std
        self.estados = torch.from_numpy(estados)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Dataset: {len(self.estados)-k_steps} secuencias, dt={self.dt}, k_steps={k_steps}")

    def __len__(self):
        return len(self.estados) - self.k_steps

    def __getitem__(self, idx):
        x = self.estados[idx]
        y_seq = self.estados[idx+1:idx+1+self.k_steps]
        return x, y_seq

def loss_multistep(modelo, x0, y_seq, media, std, lambda_energia=0.1):
    B, k_steps, _ = y_seq.shape
    device = x0.device
    media_t = torch.from_numpy(media).to(device)
    std_t = torch.from_numpy(std).to(device)

    x_curr = x0
    pred_seq = []
    for _ in range(k_steps):
        x_next = modelo(x_curr)
        pred_seq.append(x_next)
        x_curr = x_next
    pred_seq = torch.stack(pred_seq, dim=1)

    loss_mse = torch.mean((pred_seq - y_seq)**2)

    x0_desnorm = x0 * std_t + media_t
    pred_desnorm = pred_seq * std_t + media_t
    E0 = energia_pendulo_doble(x0_desnorm)
    E_pred = energia_pendulo_doble(pred_desnorm.reshape(-1, 4)).reshape(B, k_steps)
    loss_energia = torch.mean((E_pred - E0.unsqueeze(1))**2 / (torch.abs(E0.unsqueeze(1)) + 1e-8))

    return loss_mse + lambda_energia * loss_energia, loss_mse.detach(), loss_energia.detach()

def train_model_ddp(ruta_datos: str, hidden_dim: int = 128, n_layers: int = 3,
                    epocas: int = 500, batch_size: int = 128, lr: float = 1e-3,
                    ruta_guardado: str = "./resultados", num_workers: int = 8,
                    compile: bool = True, k_steps: int = 10, lambda_energia: float = 0.2):

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dataset = PenduloDataset(ruta_datos, k_steps=k_steps)
    n = len(dataset)
    n_train, n_val = int(0.7*n), int(0.15*n)

    g = torch.Generator().manual_seed(42)
    train_set, val_set, _ = torch.utils.data.random_split(
        dataset, [n_train, n_val, n - n_train - n_val], generator=g
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    modelo = MLP(4, 4, hidden_dim, n_layers).to(device)
    if compile:
        modelo = torch.compile(modelo, mode="reduce-overhead")

    modelo = DDP(modelo, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.Adam(modelo.parameters(), lr=lr * world_size)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    mejor_val = float('inf')
    if rank == 0:
        Path(ruta_guardado).mkdir(parents=True, exist_ok=True)

    for epoch in range(epocas):
        train_sampler.set_epoch(epoch)
        modelo.train()
        train_loss = 0.0
        train_mse = 0.0
        train_e = 0.0
        for x, y_seq in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False, disable=rank != 0):
            x, y_seq = x.to(device, non_blocking=True), y_seq.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_mse, loss_e = loss_multistep(modelo, x, y_seq, dataset.media, dataset.std, lambda_energia)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_mse += loss_mse.item()
            train_e += loss_e.item()

        modelo.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_seq in val_loader:
                x, y_seq = x.to(device, non_blocking=True), y_seq.to(device, non_blocking=True)
                loss, _, _ = loss_multistep(modelo, x, y_seq, dataset.media, dataset.std, lambda_energia)
                val_loss += loss.item()

        train_loss_t = torch.tensor(train_loss / len(train_loader)).to(device)
        val_loss_t = torch.tensor(val_loss / len(val_loader)).to(device)
        train_mse_t = torch.tensor(train_mse / len(train_loader)).to(device)
        train_e_t = torch.tensor(train_e / len(train_loader)).to(device)

        dist.all_reduce(train_loss_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(train_mse_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(train_e_t, op=dist.ReduceOp.AVG)

        scheduler.step(val_loss_t)

        if rank == 0 and epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | Train: {train_loss_t:.6e} | MSE: {train_mse_t:.6e} | E: {train_e_t:.6e} | Val: {val_loss_t:.6e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss_t < mejor_val and rank == 0:
            mejor_val = val_loss_t
            model_to_save = modelo.module._orig_mod if hasattr(modelo.module, '_orig_mod') else modelo.module
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'media': dataset.media,
                'std': dataset.std,
                'dt': dataset.dt,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'val_loss': val_loss_t.item(),
                'k_steps': k_steps,
                'lambda_energia': lambda_energia
            }, f"{ruta_guardado}/modelo_multistep.pth")

    if rank == 0:
        print(f"Mejor modelo guardado en {ruta_guardado}/modelo_multistep.pth")

def cargar_modelo_ddp(ruta_pth: str, device: str = "cuda:0", compile: bool = False):
    checkpoint = torch.load(ruta_pth, map_location=device)
    modelo = MLP(4, 4, checkpoint['hidden_dim'], checkpoint['n_layers'])
    modelo.load_state_dict(checkpoint['model_state_dict'])
    if compile and "cuda" in device:
        modelo = torch.compile(modelo, mode="reduce-overhead")
    modelo.eval()
    return modelo, checkpoint['media'], checkpoint['std'], checkpoint['dt'], checkpoint.get('k_steps', 1)