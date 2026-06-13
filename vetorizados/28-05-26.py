import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List
from tqdm import tqdm

# --- CÁLCULO VETORIZADO EM LOTE (A espinha dorsal da otimização) ---

def calculate_det_grid_vectorized(vp: float, vs: float, rho: float, h: float, 
                                  omega_arr: np.ndarray, kx_arr: np.ndarray, 
                                  G1: np.ndarray) -> np.ndarray:
    """
    Calcula os determinantes de G para toda a grade de omega e kx de forma 
    100% vetorizada usando arrays estáveis do NumPy (sem loops Python).
    """
    # Cria a grade dimensional (N_omega, N_kx) e achata para processamento linear
    O, Kx = np.meshgrid(omega_arr, kx_arr, indexing='ij')
    O = O.ravel()
    Kx = Kx.ravel()
    N = O.shape[0]

    # Constantes de Lamé
    lame_mu = (vs**2) * rho
    lame_lambda = ((vp**2) * rho) - (2 * lame_mu)

    # Números de onda (adicionando +0j para forçar suporte a complexos no sqrt)
    k_l = O / vp
    k_t = O / vs
    k_zt = np.sqrt((k_t**2 - Kx**2) + 0j)
    k_zl = np.sqrt((k_l**2 - Kx**2) + 0j)

    # Ângulos de polarização
    sin_tl = Kx / k_l
    cos_tl = k_zl / k_l
    sin_tt = Kx / k_t
    cos_tt = k_zt / k_t

    # Alocação de matrizes empilhadas (N, 3, 3)
    A1 = np.zeros((N, 3, 3), dtype=np.complex128)
    A1[:, 0, 0] = sin_tl
    A1[:, 0, 1] = -cos_tt
    A1[:, 1, 2] = 1.0
    A1[:, 2, 0] = cos_tl
    A1[:, 2, 1] = sin_tt

    A2 = A1 * np.array([[1, 1, 1], [1, 1, 1], [-1, -1, 1]])

    # Construção de L1
    L1 = np.zeros((N, 3, 3), dtype=np.complex128)
    L1[:, 0, 0] = lame_lambda * (k_zl * cos_tl + Kx * sin_tl) + 2 * lame_mu * k_zl * cos_tl
    L1[:, 0, 1] = lame_lambda * (k_zt * sin_tt - Kx * cos_tt) + 2 * lame_mu * k_zt * sin_tt
    L1[:, 1, 0] = lame_mu * (Kx * cos_tl + k_zl * sin_tl)
    L1[:, 1, 1] = lame_mu * (Kx * sin_tt - k_zt * cos_tt)
    L1[:, 2, 2] = lame_mu * k_zt

    L2 = L1 * np.array([[1, 1, 1], [-1, -1, 1], [1, 1, -1]])

    # Inversão em lote das matrizes de deslocamento
    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)

    # Tensores de impedância local (Z = -(1/w) * L @ A_inv)
    inv_omega = -(1.0 / O)[:, np.newaxis, np.newaxis]
    Z1 = inv_omega * (L1 @ A1_inv)
    Z2 = inv_omega * (L2 @ A2_inv)

    # Matrizes Phi (Exponenciais diagonais)
    Phi1 = np.zeros((N, 3, 3), dtype=np.complex128)
    Phi1[:, 0, 0] = np.exp(k_zl * h * 1j)
    Phi1[:, 1, 1] = np.exp(k_zt * h * 1j)
    Phi1[:, 2, 2] = np.exp(k_zt * h * 1j)

    Phi2 = np.zeros((N, 3, 3), dtype=np.complex128)
    Phi2[:, 0, 0] = np.exp(-k_zl * h * 1j)
    Phi2[:, 1, 1] = np.exp(-k_zt * h * 1j)
    Phi2[:, 2, 2] = np.exp(-k_zt * h * 1j)

    # Matrizes M
    M1 = A1 @ Phi1 @ A1_inv
    M2Minus = A2 @ Phi1 @ A2_inv

    # R1 e W
    # G1 expandido para formato (N, 3, 3)
    G1_stack = np.repeat(G1[np.newaxis, :, :], N, axis=0)
    r1 = np.linalg.inv(Z1 - G1_stack) @ (G1_stack - Z2)
    w = M1 @ r1 @ M2Minus

    # Matriz G final e cálculo do Determinante
    I_stack = np.repeat(np.identity(3, dtype=np.complex128)[np.newaxis, :, :], N, axis=0)
    g = (Z1 @ w + Z2) @ np.linalg.inv(w + I_stack)
    
    dets = np.linalg.det(g)
    return dets.reshape(len(omega_arr), len(kx_arr))


# --- CÁLCULO UNITÁRIO PARA BISSECÇÃO ---

def calculate_single_det(vp: float, vs: float, rho: float, h: float, 
                         omega: float, kx: float, G1: np.ndarray) -> float:
    """Função auxiliar escalar rápida para a bisseção."""
    lame_mu = (vs**2) * rho
    lame_lambda = ((vp**2) * rho) - (2 * lame_mu)

    k_l, k_t = omega / vp, omega / vs
    k_zt, k_zl = np.emath.sqrt(k_t**2 - kx**2), np.emath.sqrt(k_l**2 - kx**2)

    sin_tl, cos_tl = kx / k_l, k_zl / k_l
    sin_tt, cos_tt = kx / k_t, k_zt / k_t

    A1 = np.array([[sin_tl, -cos_tt, 0j], [0j, 0j, 1+0j], [cos_tl, sin_tt, 0j]], dtype=np.complex128)
    A2 = A1 * np.array([[1, 1, 1], [1, 1, 1], [-1, -1, 1]])

    L1 = np.array([
        [lame_lambda*(k_zl*cos_tl + kx*sin_tl) + 2*lame_mu*k_zl*cos_tl, lame_lambda*(k_zt*sin_tt - kx*cos_tt) + 2*lame_mu*k_zt*sin_tt, 0j],
        [lame_mu*(kx*cos_tl + k_zl*sin_tl), lame_mu*(kx*sin_tt - k_zt*cos_tt), 0j],
        [0j, 0j, lame_mu*k_zt]
    ], dtype=np.complex128)
    L2 = L1 * np.array([[1, 1, 1], [-1, -1, 1], [1, 1, -1]])

    Z1 = -(1.0 / omega) * L1 @ np.linalg.inv(A1)
    Z2 = -(1.0 / omega) * L2 @ np.linalg.inv(A2)

    Phi1 = np.diag([np.exp(k_zl*h*1j), np.exp(k_zt*h*1j), np.exp(k_zt*h*1j)])
    M1 = A1 @ Phi1 @ np.linalg.inv(A1)
    M2Minus = A2 @ Phi1 @ np.linalg.inv(A2)

    r1 = np.linalg.inv(Z1 - G1) @ (G1 - Z2)
    w = M1 @ r1 @ M2Minus
    g = (Z1 @ w + Z2) @ np.linalg.inv(w + np.identity(3, dtype=np.complex128))
    # GYHP: Aqui tava o abs, mas n pode ficar aqui pq senão na verificação de sinal vai dar merda
    return float(np.linalg.det(g).real)


# --- TRABALHO PARALELO (BISSECÇÃO) ---

def bisection_worker(args) -> Tuple[float, float]:
    """Executa a bisseção para um par de raízes candidatas isoladas."""
    vp, vs, rho, h, omega, kx_1, kx_2, G1, tol, max_iter = args
    
    det1 = calculate_single_det(vp, vs, rho, h, omega, kx_1, G1)
    
    for _ in range(max_iter):
        kx_mid = (kx_1 + kx_2) / 2
        det_mid = calculate_single_det(vp, vs, rho, h, omega, kx_mid, G1)
        
        if abs(det_mid) < tol or ((kx_2 - kx_1) / 2) < tol: # GYHP: Coloquei o abs aqui em vez de dentro da função do det.
            return kx_mid, omega
        
        # Como estamos avaliando o módulo/comportamento do det, verificamos a variação do sinal aproximado
        if (det1 * det_mid) < 0: # GYHP: Aqui é onde ia dar merda se a função determinante já desse o valor absoluto
            kx_2 = kx_mid
        else:
            kx_1 = kx_mid
            det1 = det_mid
            
    return (kx_1 + kx_2) / 2, omega


# --- PLOTTING ---

def plotting(dados: Dict[str, List[Tuple[float, float]]], x_lbl: str, y_lbl: str, tol: float, exec_time: float):
    plt.figure(figsize=(10, 6))
    for grupo, pontos in dados.items():
        if pontos:
            x = [p[0] for p in pontos]
            y = [p[1] for p in pontos]
            plt.scatter(x, y, c='black', marker='.', alpha=0.8, s=1)

    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(f'Tolerance: {tol} | Vetorizado + Paralelizado\nExecution Time: {exec_time:.4f} sec')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'tolerance_{tol}.png', dpi=300, bbox_inches='tight')


# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    medium_data = {'2': {'vp': 3.0, 'vs': 1.5, 'rho': 7.0, 'h': 1.0}}
    G1 = np.zeros((3, 3), dtype=np.complex128)
    # THRESHOLD = 1e-6
    THRESHOLD = 1e1
    TOL_BISECTION = 1e-8
    MAX_ITER = 100
    
    DEBUG = True
    
    # if DEBUG:
    #     MIN_OMEGA = 1e-3
    #     MAX_OMEGA = 38.0
    #     DELTA_OMEGA = 1.0
    #     NUM_OMEGAS = int((MAX_OMEGA - MIN_OMEGA) // DELTA_OMEGA) + 1
    #     REAL_LAST_OMEGA = MIN_OMEGA + (NUM_OMEGAS - 1) * DELTA_OMEGA
    #
    #     MIN_K = 1e-3
    #     MAX_K =  22
    #     DELTA_K = 22*1e-2
    #     NUM_KS = int((MAX_K - MIN_K) // DELTA_K) + 1
    #     REAL_LAST_K = MIN_K + (NUM_KS - 1) * DELTA_K
    # else:
    #     # omega do código do professor com guilherme
    #     # 1e-4:2.5e-2:14
    #     MIN_OMEGA = 1e-4
    #     MAX_OMEGA = 14.0
    #     DELTA_OMEGA = 2.5e-2
    #     NUM_OMEGAS = int((MAX_OMEGA - MIN_OMEGA) // DELTA_OMEGA) + 1
    #     REAL_LAST_OMEGA = MIN_OMEGA + (NUM_OMEGAS - 1) * DELTA_OMEGA
    #     # kx do código do professor com guilherme
    #     # 0.000001:0.0000001:0.001
    #     MIN_K = 1e-6
    #     MAX_K = 1e-3
    #     # DELTA_K = 1e-7
    #     DELTA_K = 1e-5
    #     NUM_KS = int((MAX_K - MIN_K) // DELTA_K) + 1
    #     REAL_LAST_K = MIN_K + (NUM_KS - 1) * DELTA_K
    #
    # Configuração de Grade Ampla (Modo Produção)
    # omegas = np.linspace(MIN_OMEGA, REAL_LAST_OMEGA, NUM_OMEGAS)
    # ks = np.linspace(MIN_K, REAL_LAST_K, NUM_KS)

    # GYHP: Overwrite do código original dos parametros
    MIN_OMEGA = 1e-4
    MAX_OMEGA = 36
    NUM_OMEGAS = 600

    MIN_K = 0.0
    MAX_K = 15.0
    NUM_KS = 1000

    omegas = np.linspace(MIN_OMEGA, MAX_OMEGA, NUM_OMEGAS)
    ks = np.linspace(MIN_K, MAX_K, NUM_KS)
    # GYHP: Overwrite do código original dos parametros


    print(f"Iniciando varredura de grade vetorizada ({len(omegas)}x{len(ks)})...")
    inicio = time.perf_counter()

    # 1. Varredura da Grade Inteira Vetorizada (Substitui o loop massivo original)
    grid_dets = calculate_det_grid_vectorized(
        vp=medium_data['2']['vp'], vs=medium_data['2']['vs'], rho=medium_data['2']['rho'], h=medium_data['2']['h'],
        omega_arr=omegas, kx_arr=ks, G1=G1
    )
    
    # Obter os valores absolutos reais para análise de Bolzano / Sinais
    grid_dets_real = np.real(grid_dets)

    # 2. Identificar cruzamentos de zero/mudanças de comportamento na grade (Utilizando shifts do NumPy)
    # Verifica mudanças de sinal entre colunas adjacentes (kx adjacentes para o mesmo omega)
    sign_changes = (grid_dets_real[:, :-1] * grid_dets_real[:, 1:]) < 0
    
    # Filtragem por Threshold de proximidade absoluta
    # threshold_mask = np.abs(np.abs(grid_dets_real[:, :-1]) - np.abs(grid_dets_real[:, 1:])) < THRESHOLD
    # threshold_mask = np.abs(grid_dets_real[:, :-1]) < THRESHOLD # GYHP: Buscando apenas por pontos próximos a zero já
    threshold_mask = np.abs(grid_dets[:, :-1]) < THRESHOLD # GYHP: Dif só do ponto, sem comparação vizinha e grid_dets não grid_dets_real!!!
    candidates_mask = sign_changes & threshold_mask
    
    # Encontra os índices onde as condições foram satisfeitas
    omega_idxs, kx_idxs = np.where(candidates_mask)
    
    # # 3. Preparar pacotes de dados para bisseção paralela nos candidatos encontrados
    # bisection_tasks = []
    # for o_idx, k_idx in zip(omega_idxs, kx_idxs):
    #     bisection_tasks.append((
    #         medium_data['2']['vp'], medium_data['2']['vs'], medium_data['2']['rho'], medium_data['2']['h'],
    #         omegas[o_idx], ks[k_idx], ks[k_idx + 1], G1, TOL_BISECTION, MAX_ITER
    #     ))
    #
    # print(f"Encontrados {len(bisection_tasks)} candidatos a raiz. Iniciando Bisseção Paralela...")

    # # 4. Paralelização Multi-Core real para Processamento da Bisseção
    # found_omega_kx = {}
    # if bisection_tasks:
    #     with ProcessPoolExecutor() as executor:
    #         # Mapeia as tarefas entre os núcleos disponíveis do processador
    #         results = list(tqdm(executor.map(bisection_worker, bisection_tasks), total=len(bisection_tasks), desc="Executando Bisseção"))
    #
    #     # Agrupa os resultados encontrados por Ômega para manter compatibilidade com sua função de plot original
    #     for kx_res, omega_res in results:
    #         key = f"{omega_res}"
    #         if key not in found_omega_kx:
    #             found_omega_kx[key] = []
    #         found_omega_kx[key].append((kx_res, omega_res))

    # A não-bisseção pra só pegar o ponto médio msm.
    found_omega_kx = {}
    for o_idx, k_idx in zip(omega_idxs, kx_idxs):
        kx_mid = (ks[k_idx] + ks[k_idx + 1]) / 2
        omega_val = omegas[o_idx]
        key = f"{omega_val}"
        if key not in found_omega_kx:
            found_omega_kx[key] = []
        found_omega_kx[key].append((kx_mid, omega_val))

    fim = time.perf_counter()
    tempo_execucao = fim - inicio
    print(f"Concluído com sucesso em {tempo_execucao:.4f} segundos!")

    # Plotar resultados
    KX_LBL = f"KX min: {ks[0]:.4f} max: {ks[-1]:.4f} len: {len(ks)}"
    OMEGA_LBL = f"OMEGA min: {omegas[0]:.4f} max: {omegas[-1]:.4f} len: {len(omegas)}"
    plotting(dados=found_omega_kx, x_lbl=KX_LBL, y_lbl=OMEGA_LBL, exec_time=tempo_execucao, tol=THRESHOLD)
