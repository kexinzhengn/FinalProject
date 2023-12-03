import os,glob 
import numpy as np
import open3d as o3d
from operator import itemgetter
import random

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                        'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])

CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}



def generate_colors(composite_labels, random_color=True):
    '''
    Semantic types outside of the 20 classes are ignored.
    '''
    
    sem_pred = np.floor(composite_labels / 1000)
    sem_pred = sem_pred.astype(np.int32)
    instances = composite_labels - sem_pred * 1000
    N = sem_pred.shape[0]
    
    # Semantic color
    valid = (sem_pred>=0) & (sem_pred<20)
    
    if valid.sum() < 1:
        return None, None
    semantic_colors = np.zeros((N, 3)).astype(np.uint8)
    semantic_colors[valid] = np.array(itemgetter(*SEMANTIC_NAMES[sem_pred[valid]])(CLASS_COLOR))

    # Instance color
    instance_colors = np.zeros((N, 3)).astype(np.int32)
    instance_list = np.unique(instances)
    for idx in instance_list[:-1]:
        if random_color:
            instance_colors[instances == idx] = np.random.randint(0, 256, size=3)
        else:
            instance_colors[instances == idx] = COLOR20[idx % len(COLOR20)]
        
    return semantic_colors, instance_colors

# todo
def create_instance_centroid(points, instances, labels):
    instance_idxs = np.unique(instances)
    centroid_spheres = []
    for id in instance_idxs:
        instance_points = points[instances==id]
        centroid = np.mean(instance_points,axis=0)
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        label_name = SEMANTIC_NAMES[labels[instances==id][0]]
        label_color = CLASS_COLOR[label_name]
        # mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([label_color[0] / 255.0,label_color[1]/255.0,label_color[2]/255.0]) # ([0.1, 0.1, 0.7])
        centroid_spheres.append(mesh_sphere.translate(centroid))
    return centroid_spheres

