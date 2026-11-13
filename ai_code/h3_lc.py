import os, sys, inspect, torch
sys.path.insert(0, r"D:\ai\py_h3")
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
print("=== load_components 签名 ===")
print(inspect.signature(MiniMaxH3ModularPipeline.load_components))
d = (MiniMaxH3ModularPipeline.load_components.__doc__ or "")[:900]
print("=== docstring ===")
print(d)
print("=== get_component_spec 签名 ===")
print(inspect.signature(MiniMaxH3ModularPipeline.get_component_spec))
os._exit(0)