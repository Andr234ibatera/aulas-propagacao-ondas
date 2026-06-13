import torch
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def calculate_det_grid_torch(vp: float, vs: float, rho: float, h: float, 
                             omega_arr: torch.Tensor, kx_arr: torch.Tensor, 
                             G1: torch.Tensor) -> torch.Tensor:
    """Calcula os determinantes de G para toda a grade usando tensores PyTorch.
    Altamente otimizado para execução paralela em GPU e uso mínimo de memória.
    """
    device = omega_arr.device
    
    # Cria a grade dimensional e achata para processamento linear
    O, Kx = torch.meshgrid(omega_arr, kx_arr, indexing='ij')
    O = O.flatten()
    Kx = Kx.flatten()
    N = O.shape[0]

    # Constantes de Lamé
    lame_mu = (vs**2) * rho
    lame_lambda = ((vp**2) * rho) - (2 * lame_mu)

    # Números de onda (conversão para complex128 para suportar raízes negativas)
    k_l = (O / vp).to(torch.complex128)
    k_t = (O / vs).to(torch.complex128)
    Kx_c = Kx.to(torch.complex128)

    k_zt = torch.sqrt(k_t**2 - Kx_c**2)
    k_zl = torch.sqrt(k_l**2 - Kx_c**2)

    # Ângulos de polarização
    sin_tl = Kx_c / k_l
    cos_tl = k_zl / k_l
    sin_tt = Kx_c / k_t
    cos_tt = k_zt / k_t

    # Alocação de matrizes A1
    A1 = torch.zeros((N, 3, 3), dtype=torch.complex128, device=device)
    A1[:, 0, 0] = sin_tl
    A1[:, 0, 1] = -cos_tt
    A1[:, 1, 2] = 1.0
    A1[:, 2, 0] = cos_tl
    A1[:, 2, 1] = sin_tt

    mult_A = torch.tensor([[1, 1, 1], [1, 1, 1], [-1, -1, 1]], dtype=torch.complex128, device=device)
    A2 = A1 * mult_A

    # Alocação de L1
    L1 = torch.zeros((N, 3, 3), dtype=torch.complex128, device=device)
    L1[:, 0, 0] = lame_lambda * (k_zl * cos_tl + Kx_c * sin_tl) + 2 * lame_mu * k_zl * cos_tl
    L1[:, 0, 1] = lame_lambda * (k_zt * sin_tt - Kx_c * cos_tt) + 2 * lame_mu * k_zt * sin_tt
    L1[:, 1, 0] = lame_mu * (Kx_c * cos_tl + k_zl * sin_tl)
    L1[:, 1, 1] = lame_mu * (Kx_c * sin_tt - k_zt * cos_tt)
    L1[:, 2, 2] = lame_mu * k_zt

    mult_L = torch.tensor([[1, 1, 1], [-1, -1, 1], [1, 1, -1]], dtype=torch.complex128, device=device)
    L2 = L1 * mult_L

    # Inversão em lote
    A1_inv = torch.linalg.inv(A1)
    A2_inv = torch.linalg.inv(A2)

    # Tensores de impedância local
    inv_omega = -(1.0 / O.to(torch.complex128)).unsqueeze(1).unsqueeze(2)
    Z1 = inv_omega * torch.matmul(L1, A1_inv)
    Z2 = inv_omega * torch.matmul(L2, A2_inv)

    # Matrizes Phi
    Phi1 = torch.zeros((N, 3, 3), dtype=torch.complex128, device=device)
    Phi1[:, 0, 0] = torch.exp(k_zl * h * 1j)
    Phi1[:, 1, 1] = torch.exp(k_zt * h * 1j)
    Phi1[:, 2, 2] = torch.exp(k_zt * h * 1j)

    # Matrizes M
    M1 = torch.matmul(A1, torch.matmul(Phi1, A1_inv))
    M2Minus = torch.matmul(A2, torch.matmul(Phi1, A2_inv))

    # G1_stack usando expand() para O(1) de custo de memória
    G1_stack = G1.unsqueeze(0).expand(N, 3, 3)

    r1 = torch.matmul(torch.linalg.inv(Z1 - G1_stack), (G1_stack - Z2))
    w = torch.matmul(M1, torch.matmul(r1, M2Minus))

    # I_stack expandido (O(1) memória)
    I_stack = torch.eye(3, dtype=torch.complex128, device=device).unsqueeze(0).expand(N, 3, 3)
    
    g = torch.matmul((torch.matmul(Z1, w) + Z2), torch.linalg.inv(w + I_stack))
    
    # Determinante em lote
    dets = torch.linalg.det(g)
    return dets.reshape(len(omega_arr), len(kx_arr))

if __name__ == "__main__":
    # Configuração de Dispositivo (GPU se disponível, senão CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # Para Macs M1/M2/M3
    else:
        device = torch.device('cpu')
        
    print(f"Executando no dispositivo: {device}")

    medium_data = {'2': {'vp': 3.0, 'vs': 1.5, 'rho': 7.0, 'h': 1.0}}
    G1 = torch.zeros((3, 3), dtype=torch.complex128, device=device)
    
    THRESHOLD = 1e-3
    MIN_OMEGA, MAX_OMEGA, NUM_OMEGAS = 1e-4, 36.0, 800
    MIN_K, MAX_K, NUM_KS = 0.0, 15.0, 10000

    # Criação dos tensores de grade já no dispositivo alvo
    omegas = torch.linspace(MIN_OMEGA, MAX_OMEGA, NUM_OMEGAS, device=device)
    ks = torch.linspace(MIN_K, MAX_K, NUM_KS, device=device)

    print(f"Iniciando varredura de grade otimizada ({len(omegas)}x{len(ks)})...")
    inicio = time.perf_counter()

    # Cálculo vetorizado e acelerado
    grid_dets = calculate_det_grid_torch(
        vp=medium_data['2']['vp'], vs=medium_data['2']['vs'], 
        rho=medium_data['2']['rho'], h=medium_data['2']['h'],
        omega_arr=omegas, kx_arr=ks, G1=G1
    )
    
    # Processamento puramente em tensores (evita gargalos com a CPU)
    grid_dets_real = torch.real(grid_dets)

    # Identificar cruzamentos de zero/mudanças de comportamento na grade
    sign_changes = (grid_dets_real[:, :-1] * grid_dets_real[:, 1:]) < 0
    
    # Filtragem por Threshold
    threshold_mask = torch.abs(grid_dets[:, :-1]) < THRESHOLD
    candidates_mask = sign_changes & threshold_mask
    
    # Traz os índices para a CPU apenas no momento final
    omega_idxs, kx_idxs = torch.where(candidates_mask)
    
    # O loop abaixo roda na CPU, pois a extração em dicionário exige manipulação Python
    found_omega_kx = {}
    ks_cpu = ks.cpu().numpy()
    omegas_cpu = omegas.cpu().numpy()
    
    for o_idx, k_idx in zip(omega_idxs.cpu().tolist(), kx_idxs.cpu().tolist()):
        kx_mid = (ks_cpu[k_idx] + ks_cpu[k_idx + 1]) / 2
        omega_val = omegas_cpu[o_idx]
        key = f"{omega_val}"
        if key not in found_omega_kx:
            found_omega_kx[key] = []
        found_omega_kx[key].append((kx_mid, omega_val))

    fim = time.perf_counter()
    tempo_execucao = fim - inicio
    print(f"Concluído com sucesso em {tempo_execucao:.4f} segundos!")

    # Plotar resultados (matplotlib precisa de numpy/CPU)
    det_mag = torch.abs(grid_dets).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(ks_cpu, omegas_cpu, det_mag, norm=LogNorm(vmin=det_mag.min(), vmax=det_mag.max()), shading="auto", cmap="bwr")
    plt.title(f"Tolerance: {THRESHOLD} | Exec. time: {tempo_execucao:.4f} secs\n(Kx:Omega)={NUM_KS}x{NUM_OMEGAS} | Device: {device}")
    plt.colorbar(label="abs(det(G))")
    plt.xlabel(f"Kx min: {MIN_K} max: {MAX_K} qtt: {NUM_KS}")
    plt.ylabel(f"Omega min: {MIN_OMEGA} max: {MAX_OMEGA} qtt: {NUM_OMEGAS}")
    plt.ylim(0, MAX_OMEGA)
    plt.tight_layout()
    plt.savefig(f'pcolor_abs_det_torch_{NUM_KS}_{NUM_OMEGAS}_{str(time.ctime(time.time())).replace(":","_").replace(" ","_")}', dpi=300, bbox_inches="tight")