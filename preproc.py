from sys import argv
from os.path import exists
import open3d as opd
import pymeshlab as pm
import numpy as np

if len(argv) < 3:
 print("Not enough arguments. Need input fle and output filename")
 exit(1)

input_file = argv[1]
output_file = argv[2]

if not exists(input_file):
 print(f"File {input_file} does not exist")
 exit(1)

ms = pm.MeshSet()
ms.load_new_mesh(input_file)
ms.save_current_mesh(output_file, save_textures=False)

mesh = opd.io.read_triangle_mesh(output_file)
mesh.remove_non_manifold_edges()
mesh.remove_unreferenced_vertices()
opd.io.write_triangle_mesh(output_file, mesh)

print(f"Processed file {input_file} to {output_file}")
