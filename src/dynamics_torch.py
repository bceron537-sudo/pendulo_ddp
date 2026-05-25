import torch

def energia_pendulo_doble(estado, g=9.81, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    """Energía total. estado: (..., 4) = [th1, w1, th2, w2]"""
    theta1, omega1, theta2, omega2 = torch.unbind(estado, dim=-1)
    y1 = -L1 * torch.cos(theta1)
    y2 = y1 - L2 * torch.cos(theta2)
    V = m1*g*y1 + m2*g*y2
    v1_sq = (L1*omega1)**2
    v2_sq = (L1*omega1)**2 + (L2*omega2)**2 + 2*L1*L2*omega1*omega2*torch.cos(theta1-theta2)
    T = 0.5*m1*v1_sq + 0.5*m2*v2_sq
    return T + V

def symplectic_euler_pendulo(estado, dt, g=9.81, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    theta1, omega1, theta2, omega2 = torch.unbind(estado, dim=-1)
    delta = theta1 - theta2
    den1 = (m1+m2)*L1 - m2*L1*torch.cos(delta)**2
    den2 = (L2/L1)*den1
    a1 = (m2*L1*omega1**2*torch.sin(delta)*torch.cos(delta)
          + m2*g*torch.sin(theta2)*torch.cos(delta)
          + m2*L2*omega2**2*torch.sin(delta)
          - (m1+m2)*g*torch.sin(theta1)) / den1
    a2 = (-m2*L2*omega2**2*torch.sin(delta)*torch.cos(delta)
          + (m1+m2)*g*torch.sin(theta1)*torch.cos(delta)
          - (m1+m2)*L1*omega1**2*torch.sin(delta)
          - (m1+m2)*g*torch.sin(theta2)) / den2
    omega1_new = omega1 + a1 * dt
    omega2_new = omega2 + a2 * dt
    theta1_new = theta1 + omega1_new * dt
    theta2_new = theta2 + omega2_new * dt
    return torch.stack([theta1_new, omega1_new, theta2_new, omega2_new], dim=-1)