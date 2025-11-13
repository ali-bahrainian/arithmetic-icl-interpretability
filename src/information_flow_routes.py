import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm.auto import tqdm

from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.routes.graph import GraphBuilder, build_full_graph, build_paths_to_predictions


OPERATORS = {'+', '-', '*', '/', 'times', 'minus', 'plus', 'divide'}
EQUAL_SIGNS = {'=', 'is', 'equals'}


def generate_graphs(model, dataset, results, threshold=0.04):
    # TODO: write docstring, improve codestyle, add comments
    contribution_graphs = []

    for i, item in enumerate(tqdm(dataset)):
        if results[i]:
            msg = item[0]
            model.run(msg)

            graph = build_full_graph(model, renormalizing_threshold=threshold)

            tokens = model.tokens()[0]
            n_tokens = tokens.shape[0]
            model_info = model.model_info()
            paths = build_paths_to_predictions(
                graph,
                model_info.n_layers,
                n_tokens,
                range(n_tokens),
                threshold,
            )
            
            contribution_graphs.append((graph.copy(), paths[-1], tokens.cpu().numpy()))
    return contribution_graphs
    

def take_operators_operands_nodes_from_contribution_graphs(contribution_graphs, tokenizer):
    # TODO: write docstring, improve codestyle, add comments

    contribution_graphs_operands_operators = []

    for item in contribution_graphs:
        # Extract the graph, path, and tokens
        graph, path, tokens = item[0].copy(), item[1].copy(), item[2]
        
        # Identify positions of numeric operands in the tokens
        operands_operators_positions = []
        for i, token in enumerate(tokens):
            token_decoded = tokenizer.decode(token)
            if token_decoded.strip() in OPERATORS or token_decoded.strip() in EQUAL_SIGNS or token_decoded.strip().isnumeric():
                    operands_operators_positions.append(i)
        
        operands_operators_positions.append(len(tokens) - 1)
        
        # Identify nodes to remove from the graph (those not corresponding to operand positions)
        nodes_to_remove = []
        for node in list(graph.nodes):
            x_pos = int(node.split('_')[1])
            if x_pos not in operands_operators_positions:
                nodes_to_remove.append(node)

        graph.remove_nodes_from(nodes_to_remove)
        path.remove_nodes_from(nodes_to_remove)
        
        contribution_graphs_operands_operators.append((graph, path))
    return contribution_graphs_operands_operators
    

def set_node_coordinates(graph):
    # TODO: write docstring, improve codestyle, add comments
    all_x_pos = [int(node.split('_')[1]) for node in graph.nodes]
    sorted_pos = sorted(set(all_x_pos))
    number_to_position = {num: idx for idx, num in enumerate(sorted_pos)}
    
    positions = {}
    for node in graph.nodes:
        x_pos =  number_to_position[int(node.split('_')[1])]
        y_pos = int(node.split('_')[0][1:])
        label = node.split('_')[0][0]
        if label == 'A':
            y_pos -= 0.3
        elif label == 'M':
            x_pos += 0.3
        elif label == 'I':
            y_pos += 0.3
        elif label == 'X':
            y_pos -= 1
        positions[node] = [y_pos, -x_pos]
    return positions


def plot_information_flow_graphs(contribution_graphs, save_path):
    # TODO: write docstring, improve codestyle, add comments
    labels = ['(1) ICE 1st operand', 
        '(+) ICE 1st operator', 
        '(3) ICE 2nd operand', 
        '(+) ICE 2nd operator', 
        '(4) ICE 3rd operand', 
        '(=) ICE Equal Sign', 
        '(8) ICE Result', 
        '(2) Task 1st operand', 
        '(+) Task 1st operator', 
        '(2) Task 2nd operand', 
        '(+) Task 2nd operator', 
        '(6) Task 3rd operand', 
        '(=) Task Equal Sign']
    
    plt.figure(figsize=(8, 6))
    
    whole_graph = contribution_graphs[0][0]
    positions = set_node_coordinates(whole_graph)
    nx.draw_networkx_nodes(whole_graph, cmap="Blues", pos=positions, node_size=50, alpha=0.1)

    # Plot each individual path from the contribution graphs
    for item in tqdm(contribution_graphs):
        positions = set_node_coordinates(item[0])
        nx.draw(item[1], cmap="Blues", pos=positions, node_size=50, alpha=0.1)
        
    
    plt.axis("on")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    plt.xlabel('Layer', fontsize=16, fontname='DeJavu Serif')
    plt.title(f'Information Flow Routes Importance Graph', fontsize=16, fontname='DeJavu Serif')
    
    plt.yticks(ticks=range(-len(labels)+1, 1), labels=labels[::-1], fontsize=14, rotation=0)
    plt.xticks(fontsize=14)

    plt.gca().spines['top'].set_visible(False)   # Remove the top spine
    plt.gca().spines['right'].set_visible(False) # Remove the right spine
    plt.gca().spines['left'].set_visible(False)  # Remove the left spine
    plt.gca().spines['bottom'].set_visible(False) # Remove the bottom spine
    
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()


def average_graphs_different_thresholds(model, tokenizer, dataset, threshold_range):
    # TODO: write docstring, improve codestyle, add comments
    res = []
    for i, item in enumerate(tqdm(dataset)):
        model.run(item[0])
        tokens = model.tokens()[0]
        n_tokens = tokens.shape[0]
        model_info = model.model_info()
        node_activation_dict = defaultdict(int)
        for threshold in threshold_range:
            graph = build_full_graph(model, renormalizing_threshold=threshold)
            path = build_paths_to_predictions(
                graph,
                model_info.n_layers,
                n_tokens,
                range(n_tokens),
                threshold,
            )[-1]
            
            operands_operators_positions = []
            for i, token in enumerate(tokens):
                token_decoded = tokenizer.decode(token)
                if token_decoded.strip() in OPERATORS or token_decoded.strip() in EQUAL_SIGNS or token_decoded.strip().isnumeric():
                    operands_operators_positions.append(i)
            
            if len(tokens)-1 not in operands_operators_positions:
                operands_operators_positions.append(len(tokens) - 1)
            
            # Identify nodes to remove from the graph (those not corresponding to operand positions)
            nodes_to_remove = []
            for node in list(graph.nodes):
                x_pos = int(node.split('_')[1])
                if x_pos not in operands_operators_positions:
                    nodes_to_remove.append(node)
            
            graph.remove_nodes_from(nodes_to_remove)
            path.remove_nodes_from(nodes_to_remove)
            
            operands_operators_positions_correspondance = {operands_operators_positions[i]: i for i in range(len(operands_operators_positions))}
            nodes_correspondance = dict()
            for node in graph:
                x_pos = int(node.split('_')[1])
                x_pos_new = operands_operators_positions_correspondance[x_pos]
                node_new = node.split('_')[0] + '_' + str(x_pos_new)
                nodes_correspondance[node] = node_new
                
            graph = nx.relabel_nodes(graph, nodes_correspondance)
            path = nx.relabel_nodes(path, nodes_correspondance)
            
            for node in path:
                node_activation_dict[node] = max(node_activation_dict[node], threshold)
        res.append((graph, path, node_activation_dict))
    return res


def plot_all_graphs_different_thresholds(graphs, save_path=None):
    # TODO: write docstring, improve codestyle, add comments
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a dataframe from the matrix
    labels = ['(1) ICE 1st operand', 
        '(+) ICE 1st operator', 
        '(3) ICE 2nd operand', 
        '(+) ICE 2nd operator', 
        '(4) ICE 3rd operand', 
        '(=) ICE Equal Sign', 
        '(8) ICE Result', 
        '(2) Task 1st operand', 
        '(+) Task 1st operator', 
        '(2) Task 2nd operand', 
        '(+) Task 2nd operator', 
        '(6) Task 3rd operand', 
        '(=) Task Equal Sign']

    node_activation_all = defaultdict(list)
    for item in graphs:
        nodes_correspondance_curr = item[-1]
        for key in nodes_correspondance_curr:
            node_activation_all[key].append(nodes_correspondance_curr[key])
            
    for key in node_activation_all:
        node_activation_all[key] = np.mean(node_activation_all[key])
    
    print(node_activation_all)

    graph = graphs[0][0]
    node_colors = [node_activation_all[node] if node_activation_all[node] else 0 for node in graph]

    cmap = plt.cm.Blues
    norm = Normalize(vmin=min(node_colors), vmax=max(node_colors))

    positions = set_positions_transposed(graph)
    nx.draw_networkx_nodes(graph, node_color=node_colors, pos=positions, cmap=cmap, node_size=50)

    # Add the colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label('$\\tau$', fontsize=16)

    plt.axis("on")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.xlabel('Layer', fontsize=16, fontname='DeJavu Serif')
    plt.title(f'Relationship Between $\\tau$ and Important Components', fontsize=16, fontname='DeJavu Serif')

    # plt.xticks(range(32))
    plt.yticks(ticks=range(-len(labels)+1, 1), labels=labels[::-1], fontsize=14, rotation=0)
    plt.xticks(fontsize=14)

    plt.gca().spines['top'].set_visible(False)   # Remove the top spine
    plt.gca().spines['right'].set_visible(False) # Remove the right spine
    plt.gca().spines['left'].set_visible(False)  # Remove the left spine
    plt.gca().spines['bottom'].set_visible(False) # Remove the bottom spine
    
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches='tight')