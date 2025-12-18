import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

# 漂亮的元素颜色映射 (参考 LithiumVision)
ELEMENT_COLORS = {
    'Pt': '#D0D0E0', 'Pd': '#006985', 'Cu': '#C88033', 'Ni': '#50D050',
    'Au': '#FFD123', 'Ag': '#C0C0C0', 'Fe': '#E06633', 'Co': '#F090A0'
}
DEFAULT_COLOR = '#CCCCCC'

# 元素半径比例
ELEMENT_RADII = {
    'Pt': 1.39, 'Pd': 1.37, 'Cu': 1.28, 'Ni': 1.24,
    'Au': 1.44, 'Ag': 1.44
}
DEFAULT_RADIUS = 1.2

logger = logging.getLogger(__name__)

def plot_structure_plotly(atoms, title, output_file, width=800, height=800):
    """
    使用 Plotly 绘制 ASE Atoms 对象的高质量 3D 交互图。
    """
    if not PYMATGEN_AVAILABLE:
        logger.error("Pymatgen not installed. Cannot generate advanced visualization.")
        return False

    # 1. 转换为 Pymatgen Structure
    structure = AseAtomsAdaptor.get_structure(atoms)
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # 提取坐标
    sites = structure.sites
    coords = np.array([site.coords for site in sites])
    
    # 2. 绘制晶胞框架 (Lattice Box)
    lattice = structure.lattice
    origin = np.array([0, 0, 0])
    matrix = lattice.matrix
    a, b, c = matrix[0], matrix[1], matrix[2]
    
    points = np.array([
        origin, origin + a, origin + b, origin + c,
        origin + a + b, origin + a + c, origin + b + c,
        origin + a + b + c
    ])
    
    # 12条边
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), 
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)
    ]
    
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[points[i][0], points[j][0]],
            y=[points[i][1], points[j][1]],
            z=[points[i][2], points[j][2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='none'
        ))

    # 3. 绘制化学键 (Bonding) - 使用 CrystalNN
    try:
        # [新增] 预检查：如果原子间距太小，CrystalNN 会崩溃
        # 使用 pymatgen 的 distance matrix 快速检查
        dist_matrix = structure.distance_matrix
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)
        
        if min_dist < 1.5: # 阈值可调，小于 1.5A 肯定有问题
            logger.warning(f"Skipping bonding analysis: Atoms too close ({min_dist:.2f} A)")
        else:
            # 针对金属表面，CrystalNN 可能比较慢或连接过多，这里做一个简单的距离截断作为备选
            # 或者直接用 CrystalNN
            cnn = CrystalNN()
            graph = StructureGraph.with_local_env_strategy(structure, cnn)
            
            bond_x, bond_y, bond_z = [], [], []
            for i, j, d in graph.graph.edges(data=True):
                start = coords[i]
                end = coords[j]
                # 过滤掉跨越边界太长的键 (PBC artifacts visualization issue)
                if np.linalg.norm(start - end) < 5.0: 
                    bond_x.extend([start[0], end[0], None])
                    bond_y.extend([start[1], end[1], None])
                    bond_z.extend([start[2], end[2], None])
            
            fig.add_trace(go.Scatter3d(
                x=bond_x, y=bond_y, z=bond_z,
                mode='lines',
                line=dict(color='grey', width=3),
                opacity=0.5,
                showlegend=False,
                hoverinfo='none'
            ))
    except Exception as e:
        # [修改] 降级日志级别，避免刷屏
        logger.debug(f"Bonding analysis failed: {e}")

    # 4. 绘制原子 (Atoms)
    atom_types = [site.specie.symbol for site in sites]
    unique_atoms = set(atom_types)
    
    for atom_type in unique_atoms:
        indices = [i for i, s in enumerate(atom_types) if s == atom_type]
        atom_coords = coords[indices]
        
        color = ELEMENT_COLORS.get(atom_type, DEFAULT_COLOR)
        radius = ELEMENT_RADII.get(atom_type, DEFAULT_RADIUS)
        
        fig.add_trace(go.Scatter3d(
            x=atom_coords[:, 0],
            y=atom_coords[:, 1],
            z=atom_coords[:, 2],
            mode='markers',
            marker=dict(
                size=radius * 15, # 调整显示大小
                color=color,
                line=dict(color='black', width=1),
                opacity=1.0
            ),
            name=atom_type,
            text=[f"{atom_type} #{i}" for i in indices],
            hoverinfo='text'
        ))

    # 5. 布局设置
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.95),
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(title='X (Å)', showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(title='Y (Å)', showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(title='Z (Å)', showgrid=False, zeroline=False, showbackground=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.8, y=0.9)
    )
    
    fig.write_html(output_file)
    return True