import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.model_torch_ddp import cargar_modelo_ddp
from src.dynamics_torch import energia_pendulo_doble

def rollout_modelo(modelo, x0_norm, pasos: int, media, std, device):
    modelo.eval()
    estados = [x0_norm]
    with torch.no_grad():
        for _ in range(pasos):
            x_next = modelo(estados[-1])
            estados.append(x_next)
    pred_norm = torch.cat(estados, dim=0)
    pred_real = pred_norm * torch.from_numpy(std).to(device) + torch.from_numpy(media).to(device)
    return pred_real

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    modelo, media, std, dt, k_steps = cargar_modelo_ddp("resultados/modelo_multistep.pth", device=device, compile=True)
    print(f"Modelo cargado. dt={dt}, k_steps={k_steps}, device={device}")

    x_real = torch.tensor([[0.1, 0.0, -0.1, 0.0]], dtype=torch.float32)
    x_norm = (x_real - torch.from_numpy(media)) / torch.from_numpy(std)
    x_norm = x_norm.to(device)

    pasos = 100
    pred_red = rollout_modelo(modelo, x_norm, pasos, media, std, device)

    E = energia_pendulo_doble(pred_red)
    E0 = E[0]
    drift_rel = torch.mean(torch.abs(E - E0) / (torch.abs(E0) + 1e-8))
    print(f"Energía inicial: {E0.item():.6f}")
    print(f"Energía final: {E[-1].item():.6f}")
    print(f"Drift energía relativo: {drift_rel.item():.6e}")

    Path("resultados/pred").mkdir(parents=True, exist_ok=True)
    t = np.arange(pasos+1) * dt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, pred_red[:, 0].cpu(), label='Red theta1')
    ax[0].set_ylabel('theta1 [rad]')
    ax[0].legend()
    ax[1].plot(t, pred_red[:, 2].cpu(), label='Red theta2')
    ax[1].set_ylabel('theta2 [rad]')
    ax[1].set_xlabel('Tiempo [s]')
    plt.tight_layout()
    plt.savefig("resultados/pred/trayectorias.png", dpi=150)
    print("Plot guardado en resultados/pred/trayectorias.png")

if __name__ == "__main__":
    main()