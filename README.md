# 🎨 Coloração de Grafos via Dijkstra

> Implementação e análise de um algoritmo híbrido para coloração de grafos utilizando ordenação por Dijkstra seguida de coloração gulosa.

## 📋 Sumário
- [Sobre o Projeto](#-sobre-o-projeto)
- [Tecnologias](#-tecnologias)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Análises](#-análises)
- [Resultados](#-resultados)

## 🎯 Sobre o Projeto

Este projeto implementa uma abordagem híbrida para coloração de grafos, combinando:
1. Algoritmo de Dijkstra para ordenação dos vértices
2. Coloração gulosa sequencial
3. Análise detalhada de performance e qualidade

## 🛠 Tecnologias

- Python 3.8+
- NetworkX
- Matplotlib
- Seaborn
- NumPy
- Pandas

## 📦 Requisitos

```bash
pip install networkx matplotlib seaborn numpy pandas
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/Colorindo-Grafos-Dijkstra.git
cd Colorindo-Grafos-Dijkstra
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Uso

Execute o script principal:
```bash
python main.py
```

O programa irá:
1. Gerar grafos aleatórios de diferentes tamanhos
2. Aplicar o algoritmo híbrido de coloração
3. Produzir visualizações e análises
4. Salvar resultados na pasta `results/`

## 📊 Análises

O projeto realiza análises completas de:
- Tempo de execução (Dijkstra + Coloração)
- Uso de memória
- Número de cores utilizadas
- Estrutura dos grafos gerados
- Escalabilidade do algoritmo

## 📈 Resultados

Os resultados são salvos em `results/`:
- `performance_dashboard.png`: Dashboard completo de performance
- `grafo_colorido_*_vertices.png`: Visualizações dos grafos coloridos
- `relatorio_performance.txt`: Relatório detalhado da análise

### Visualizações Exemplo:

#### Dashboard de Performance
![Dashboard](results/performance_dashboard.png)

#### Exemplo de Grafo Colorido
![Grafo Exemplo](results/grafo_colorido_20_vertices.png)

## 📝 Características Principais

- Geração inteligente de grafos aleatórios
- Implementação otimizada de Dijkstra
- Coloração gulosa eficiente
- Análise estatística robusta
- Visualizações modernas e informativas
- Relatórios detalhados de performance

## 🤝 Contribuições

Contribuições são bem-vindas! Para contribuir:
1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## ✉️ Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter)

Link do Projeto: [https://github.com/seu-usuario/Colorindo-Grafos-Dijkstra](https://github.com/seu-usuario/Colorindo-Grafos-Dijkstra)
