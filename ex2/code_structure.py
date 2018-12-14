from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
 
from run_dqn_atari import *
graphviz = GraphvizOutput(output_file='trace_detail.png')