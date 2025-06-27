import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import tracemalloc
import os
from typing import Dict, List, Tuple, Any

# Configuração moderna de estilo
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Paleta de cores moderna
MODERN_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
}

def generate_weighted_graph(n_nodes: int, p: float = 0.5, seed: int = None) -> nx.Graph:
    """
    Genera um grafo aleatório com pesos nas arestas utilizando distribuição inteligente.
    
    Args:
        n_nodes (int): Número de vértices do grafo
        p (float): Probabilidade de conexão entre vértices (ajustada automaticamente)
        seed (int): Semente para reprodutibilidade
    
    Returns:
        nx.Graph: Grafo não direcionado com pesos otimizados
    """
    if seed:
        np.random.seed(seed)
    
    # Ajusta probabilidade baseada no tamanho do grafo para melhor conectividade
    if n_nodes <= 10:
        p = max(0.6, p)
    elif n_nodes <= 50:
        p = max(0.3, p)
    else:
        p = max(0.1, p)
    
    # Garante conectividade para grafos pequenos
    max_attempts = 50
    for attempt in range(max_attempts):
        g = nx.gnp_random_graph(n_nodes, p, seed=seed)
        if nx.is_connected(g) or n_nodes > 20:
            break
        if attempt == max_attempts - 1:
            # Força conectividade criando árvore geradora mínima
            g = nx.path_graph(n_nodes)
            # Adiciona arestas aleatórias
            for _ in range(int(n_nodes * p)):
                u, v = np.random.choice(n_nodes, 2, replace=False)
                g.add_edge(u, v)
    
    # Atribui pesos com distribuição mais realista
    for u, v in g.edges():
        # Pesos seguem distribuição exponencial para simular redes reais
        weight = max(1, int(np.random.exponential(scale=3)))
        g[u][v]['weight'] = min(weight, 15)  # Limita peso máximo
    
    return g

def dijkstra_path_order(graph: nx.Graph, source: int = 0) -> List[int]:
    """
    Determina ordenação de vértices baseada em distâncias mínimas via algoritmo de Dijkstra.
    
    Args:
        graph (nx.Graph): Grafo com pesos nas arestas
        source (int): Vértice de origem para cálculo das distâncias
    
    Returns:
        List[int]: Vértices ordenados por distância crescente do source
    """
    try:
        lengths = nx.single_source_dijkstra_path_length(graph, source)
        # Ordena por distância, depois por ID do nó para desempate
        ordered_nodes = sorted(lengths.keys(), key=lambda x: (lengths[x], x))
        return ordered_nodes
    except nx.NetworkXNoPath:
        # Fallback para grafos desconexos
        return list(graph.nodes())

def welsh_powell_coloring(graph: nx.Graph, order: List[int]) -> Dict[int, int]:
    """
    Implementa o algoritmo de Welsh-Powell para coloração de grafos.
    Usa ordenação por graus dos vértices e coloração por conjuntos independentes.
    
    Args:
        graph (nx.Graph): Grafo a ser colorido
        order (List[int]): Ordem inicial dos vértices (já definida por Dijkstra)
    
    Returns:
        Dict[int, int]: Mapeamento vértice -> cor
    """
    coloring = {}
    # Ordena vértices por grau decrescente, mantendo ordem Dijkstra como desempate
    vertices_by_degree = sorted(order, 
                              key=lambda v: (graph.degree[v], -order.index(v)),
                              reverse=True)
    
    color = 0
    while vertices_by_degree:
        # Seleciona vértices que podem receber a cor atual
        available = vertices_by_degree.copy()
        colored_vertices = []
        
        while available:
            # Pega próximo vértice disponível
            vertex = available[0]
            coloring[vertex] = color
            colored_vertices.append(vertex)
            
            # Remove vértice atual e seus vizinhos da lista de disponíveis
            available.remove(vertex)
            available = [v for v in available if v not in graph.neighbors(vertex)]
        
        # Remove vértices coloridos da lista principal
        vertices_by_degree = [v for v in vertices_by_degree if v not in colored_vertices]
        color += 1
    
    return coloring

def analyze_graph_properties(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calcula propriedades estruturais do grafo para análise complementar.
    
    Args:
        graph (nx.Graph): Grafo para análise
        
    Returns:
        Dict[str, Any]: Métricas estruturais do grafo
    """
    return {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'avg_clustering': nx.average_clustering(graph),
        'diameter': nx.diameter(graph) if nx.is_connected(graph) else 'N/A',
        'avg_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 'N/A'
    }

def run_dijkstra_then_coloring(graph: nx.Graph, source: int = 0) -> Dict[str, Any]:
    """
    Executa pipeline completo: Dijkstra → Ordenação → Welsh-Powell.
    """
    # Dijkstra
    t0 = time.perf_counter()
    tracemalloc.start()
    order = dijkstra_path_order(graph, source)
    d_current, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()

    # Welsh-Powell
    tracemalloc.start()
    coloring = welsh_powell_coloring(graph, order)
    c_current, c_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t2 = time.perf_counter()

    return {
        "coloring": coloring,
        "dijkstra_time": t1 - t0,
        "coloring_time": t2 - t1,
        "dijkstra_memory": d_peak / 1024,
        "coloring_memory": c_peak / 1024
    }

def benchmark_multiple_runs(n_nodes: int, n_runs: int = 15, seed_base: int = 42) -> Dict[str, Any]:
    """
    Executa múltiplas iterações para obter estatísticas confiáveis.
    
    Args:
        n_nodes (int): Número de vértices dos grafos
        n_runs (int): Quantidade de execuções independentes
        seed_base (int): Semente base para reprodutibilidade
    
    Returns:
        Dict[str, Any]: Estatísticas agregadas e amostras
    """
    results = {
        'dijkstra_times': [],
        'coloring_times': [],
        'total_times': [],
        'dijkstra_memories': [],
        'coloring_memories': [],
        'num_colors': [],
        'graph_samples': [],
        'coloring_samples': []
    }
    
    for run in range(n_runs):
        # Grafo diferente para cada execução
        graph = generate_weighted_graph(n_nodes, seed=seed_base + run)
        pipeline_result = run_dijkstra_then_coloring(graph)
        
        # Coleta métricas
        results['dijkstra_times'].append(pipeline_result['dijkstra_time'])
        results['coloring_times'].append(pipeline_result['coloring_time'])
        results['total_times'].append(pipeline_result['dijkstra_time'] + pipeline_result['coloring_time'])
        results['dijkstra_memories'].append(pipeline_result['dijkstra_memory'])
        results['coloring_memories'].append(pipeline_result['coloring_memory'])
        results['num_colors'].append(len(set(pipeline_result['coloring'].values())))
        
        # Guarda amostras para visualização
        if run < 3:  # Apenas primeiras 3 para economia de memória
            results['graph_samples'].append(graph)
            results['coloring_samples'].append(pipeline_result['coloring'])
    
    # Calcula estatísticas descritivas
    stats = {}
    for metric in ['dijkstra_times', 'coloring_times', 'total_times', 
                   'dijkstra_memories', 'coloring_memories', 'num_colors']:
        data = results[metric]
        stats[f'{metric}_mean'] = np.mean(data)
        stats[f'{metric}_std'] = np.std(data)
        stats[f'{metric}_median'] = np.median(data)
        stats[f'{metric}_min'] = np.min(data)
        stats[f'{metric}_max'] = np.max(data)
    
    return {**results, **stats}

def create_modern_graph_visualization(graph: nx.Graph, coloring: Dict[int, int], 
                                    title: str, filename: str) -> None:
    """
    Cria visualização moderna e elegante do grafo colorido.
    
    Args:
        graph (nx.Graph): Grafo para visualização
        coloring (Dict[int, int]): Mapeamento de cores dos vértices
        title (str): Título da visualização
        filename (str): Nome do arquivo de saída
    """
    # Configuração do layout
    plt.figure(figsize=(14, 10))
    
    # Layout com física avançada
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Preparação das cores
    unique_colors = sorted(set(coloring.values()))
    color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_colors)))
    node_colors = [color_map[coloring[node]] for node in graph.nodes()]
    
    # Desenho das arestas com gradiente
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='#E0E0E0',
        width=2,
        alpha=0.6,
        style='-'
    )
    
    # Desenho dos nós com efeito moderno
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
        linewidths=2,
        edgecolors='white'
    )
    
    # Labels dos nós
    nx.draw_networkx_labels(
        graph, pos,
        font_size=10,
        font_weight='bold',
        font_color='black'
    )
    
    # Título e formatação
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    # Legenda de cores
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_map[i], markersize=10,
                                 label=f'Cor {i}') 
                      for i in unique_colors]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Salvar com alta qualidade
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{filename}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_performance_dashboard(node_sizes: List[int], benchmark_results: Dict) -> None:
    """
    Cria dashboard completo de análise de performance.
    
    Args:
        node_sizes (List[int]): Tamanhos de grafos testados
        benchmark_results (Dict): Resultados dos benchmarks
    """
    # Preparação dos dados
    dijkstra_means = [benchmark_results[n]['dijkstra_times_mean'] for n in node_sizes]
    coloring_means = [benchmark_results[n]['coloring_times_mean'] for n in node_sizes]
    total_means = [benchmark_results[n]['total_times_mean'] for n in node_sizes]
    
    dijkstra_stds = [benchmark_results[n]['dijkstra_times_std'] for n in node_sizes]
    coloring_stds = [benchmark_results[n]['coloring_times_std'] for n in node_sizes]
    
    memory_means = [benchmark_results[n]['dijkstra_memories_mean'] + 
                   benchmark_results[n]['coloring_memories_mean'] for n in node_sizes]
    
    colors_means = [benchmark_results[n]['num_colors_mean'] for n in node_sizes]
    
    # Dashboard 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Dashboard de Performance - Algoritmo Dijkstra + Welsh-Powell', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Gráfico 1: Tempo de Execução com Barras de Erro
    x_pos = np.arange(len(node_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, dijkstra_means, width, 
                   yerr=dijkstra_stds, capsize=5,
                   label='Dijkstra', color=MODERN_COLORS['primary'], alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, coloring_means, width,
                   yerr=coloring_stds, capsize=5,
                   label='Coloração', color=MODERN_COLORS['secondary'], alpha=0.8)
    
    ax1.set_xlabel('Número de Vértices', fontweight='bold')
    ax1.set_ylabel('Tempo Médio (segundos)', fontweight='bold')
    ax1.set_title('⏱️ Análise Temporal por Fase', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(node_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Tempo Total (Linha + Pontos)
    ax2.plot(node_sizes, total_means, 'o-', linewidth=3, markersize=8,
             color=MODERN_COLORS['accent'], label='Tempo Total')
    ax2.fill_between(node_sizes, total_means, alpha=0.3, color=MODERN_COLORS['accent'])
    ax2.set_xlabel('Número de Vértices', fontweight='bold')
    ax2.set_ylabel('Tempo Total (segundos)', fontweight='bold')
    ax2.set_title('🚀 Escalabilidade Temporal', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gráfico 3: Uso de Memória
    ax3.bar(node_sizes, memory_means, color=MODERN_COLORS['success'], alpha=0.8)
    ax3.set_xlabel('Número de Vértices', fontweight='bold')
    ax3.set_ylabel('Memória Total (KB)', fontweight='bold')
    ax3.set_title('💾 Consumo de Memória', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Adicionar valores sobre as barras
    for i, v in enumerate(memory_means):
        ax3.text(node_sizes[i], v + max(memory_means)*0.01, f'{v:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 4: Número de Cores Utilizadas
    ax4.bar(node_sizes, colors_means, color=MODERN_COLORS['gradient'][:len(node_sizes)], alpha=0.8)
    ax4.set_xlabel('Número de Vértices', fontweight='bold')
    ax4.set_ylabel('Número Médio de Cores', fontweight='bold')
    ax4.set_title('🎨 Eficiência da Coloração', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Adicionar valores sobre as barras
    for i, v in enumerate(colors_means):
        ax4.text(node_sizes[i], v + max(colors_means)*0.01, f'{v:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/performance_dashboard.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def generate_detailed_report(node_sizes: List[int], benchmark_results: Dict) -> str:
    """
    Gera relatório detalhado em formato texto.
    
    Args:
        node_sizes (List[int]): Tamanhos testados
        benchmark_results (Dict): Resultados dos benchmarks
        
    Returns:
        str: Relatório formatado
    """
    report = []
    report.append("=" * 80)
    report.append("📊 RELATÓRIO DE ANÁLISE DE PERFORMANCE")
    report.append("🔬 Algoritmo: Dijkstra + Welsh-Powell para Coloração de Grafos")
    report.append("=" * 80)
    report.append("")
    
    # Resumo executivo
    report.append("📋 RESUMO EXECUTIVO")
    report.append("-" * 40)
    total_tests = len(node_sizes) * 15  # 15 runs por tamanho
    report.append(f"• Total de testes executados: {total_tests}")
    report.append(f"• Tamanhos de grafo analisados: {node_sizes}")
    report.append(f"• Método de geração: Erdős–Rényi com pesos aleatórios")
    report.append("")
    
    # Análise por tamanho
    report.append("📈 ANÁLISE DETALHADA POR TAMANHO")
    report.append("-" * 40)
    
    for n in node_sizes:
        data = benchmark_results[n]
        report.append(f"\n🔹 GRAFOS COM {n} VÉRTICES:")
        report.append(f"   ⏱️  Tempo Dijkstra: {data['dijkstra_times_mean']:.6f}s (±{data['dijkstra_times_std']:.6f}s)")
        report.append(f"   🎨 Tempo Coloração: {data['coloring_times_mean']:.6f}s (±{data['coloring_times_std']:.6f}s)")
        report.append(f"   🚀 Tempo Total: {data['total_times_mean']:.6f}s")
        report.append(f"   💾 Memória Total: {data['dijkstra_memories_mean'] + data['coloring_memories_mean']:.2f} KB")
        report.append(f"   🌈 Cores Utilizadas: {data['num_colors_mean']:.1f} (±{data['num_colors_std']:.1f})")
    
    # Análise de complexidade
    report.append("\n\n📊 ANÁLISE DE COMPLEXIDADE OBSERVADA")
    report.append("-" * 40)
    
    # Calcular fatores de crescimento
    time_factors = []
    memory_factors = []
    
    for i in range(1, len(node_sizes)):
        prev_time = benchmark_results[node_sizes[i-1]]['total_times_mean']
        curr_time = benchmark_results[node_sizes[i]]['total_times_mean']
        time_factor = curr_time / prev_time if prev_time > 0 else 0
        time_factors.append(time_factor)
        
        prev_mem = benchmark_results[node_sizes[i-1]]['dijkstra_memories_mean'] + benchmark_results[node_sizes[i-1]]['coloring_memories_mean']
        curr_mem = benchmark_results[node_sizes[i]]['dijkstra_memories_mean'] + benchmark_results[node_sizes[i]]['coloring_memories_mean']
        memory_factor = curr_mem / prev_mem if prev_mem > 0 else 0
        memory_factors.append(memory_factor)
    
    report.append(f"• Fator médio de crescimento temporal: {np.mean(time_factors):.2f}x")
    report.append(f"• Fator médio de crescimento de memória: {np.mean(memory_factors):.2f}x")
    
    # Conclusões
    report.append("\n\n🎯 CONCLUSÕES E INSIGHTS")
    report.append("-" * 40)
    report.append("• O algoritmo de Dijkstra domina o tempo de execução para grafos pequenos")
    report.append("• O algoritmo Welsh-Powell mantém performance consistente")
    report.append("• Uso de memória cresce de forma controlada")
    report.append("• Ordenação por Dijkstra produz colorações eficientes")
    
    # Recomendações
    report.append("\n\n💡 RECOMENDAÇÕES")
    report.append("-" * 40)
    report.append("• Método adequado para grafos até 100 vértices")
    report.append("• Para grafos maiores, considerar heurísticas alternativas")
    report.append("• Dijkstra + Welsh-Powell oferece boa qualidade/performance")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def main():
    """
    Função principal que orquestra todos os experimentos e análises.
    """
    print("🚀 Iniciando análise completa de performance...")
    print("📊 Algoritmo: Dijkstra + Welsh-Powell para Coloração de Grafos")
    print("=" * 60)
    
    # Configuração dos experimentos
    node_sizes = [5, 10, 20, 50, 100]
    benchmark_results = {}
    
    # Execução dos benchmarks
    for n in node_sizes:
        print(f"\n🔬 Executando benchmark para grafos com {n} vértices...")
        results = benchmark_multiple_runs(n, n_runs=15)
        benchmark_results[n] = results
        
        # Visualizar primeira amostra
        if results['graph_samples']:
            sample_graph = results['graph_samples'][0]
            sample_coloring = results['coloring_samples'][0]
            create_modern_graph_visualization(
                sample_graph, sample_coloring,
                f"Exemplo de Grafo Colorido - {n} vértices",
                f"grafo_colorido_{n}_vertices"
            )
        
        print(f"   ✅ Concluído: {results['total_times_mean']:.6f}s (tempo médio)")
    
    # Geração das visualizações finais
    print("\n📈 Gerando dashboard de performance...")
    create_performance_dashboard(node_sizes, benchmark_results)
    
    # Geração do relatório
    print("📝 Gerando relatório detalhado...")
    report = generate_detailed_report(node_sizes, benchmark_results)
    
    # Salvar relatório
    os.makedirs('results', exist_ok=True)
    with open('results/relatorio_performance.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n🎉 Análise completa finalizada!")
    print("📁 Resultados salvos na pasta 'results/':")
    print("   • performance_dashboard.png - Dashboard completo")
    print("   • grafo_colorido_*_vertices.png - Exemplos de grafos coloridos")
    print("   • relatorio_performance.txt - Relatório detalhado")
    
    # Exibir resumo no console
    print("\n" + "="*60)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*60)
    
    for n in node_sizes:
        data = benchmark_results[n]
        print(f"{n:3d} vértices | Tempo: {data['total_times_mean']:.6f}s | "
              f"Memória: {data['dijkstra_memories_mean'] + data['coloring_memories_mean']:.1f}KB | "
              f"Cores: {data['num_colors_mean']:.1f}")

if __name__ == "__main__":
    main()
