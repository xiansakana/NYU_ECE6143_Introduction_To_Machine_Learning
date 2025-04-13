import graphviz
from graphviz import Digraph

def vis_forward_pass(variable, graph=None, parent_name=None):
    if graph is None:
        graph = Digraph(comment='Forward Pass', format='svg')
        graph.attr(rankdir="BT")
        graph.attr('node', shape='ellipse', style='filled', color='lightgreen', fontname="Helvetica")

    node_label  = f"{variable.name}\n= {variable.value:.3f}"
    graph.node(variable.name, label=node_label)

    for child_var, _ in variable.local_gradients:
        graph.edge(child_var.name, variable.name)

        vis_forward_pass(child_var, graph, parent_name=variable.name)

    return graph

def vis_backward_pass(variable, graph=None, parent_name=None):
    if graph is None:
        graph = Digraph(comment='Backward Pass', format='svg')
        graph.attr(rankdir="TB")
        graph.attr('node', shape='ellipse', style='filled', color='lightgreen', fontname="Helvetica")
        graph.attr('edge', fontname="Helvetica")


    node_label = f"{variable.name}\n \∇ {variable.accumulated_gradient:.3f}"
    graph.node(variable.name, label=node_label)

    for child_var, local_grad in variable.local_gradients:

        edge_label = f"{local_grad:.3f}"
        graph.edge(variable.name, child_var.name, label=edge_label, fontsize="12", fontcolor="grey")

        vis_backward_pass(child_var, graph, parent_name=variable.name)

    return graph
