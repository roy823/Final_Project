import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3.common.callbacks import BaseCallback
from ase.visualize.plot import plot_atoms
from ase.data import covalent_radii, atomic_numbers
from ase.io import write

# 导入新的高级可视化模块
try:
    from chem_gym.analysis.advanced_vis import plot_structure_plotly
    ADVANCED_VIS = True
except ImportError:
    ADVANCED_VIS = False

# 参考 LithiumVision 的配色方案
ELEMENT_COLORS = {
    'Pt': '#D0D0E0', 'Pd': '#006985', 'Cu': '#C88033', 'Ni': '#50D050',
    'Au': '#FFD123', 'Ag': '#C0C0C0', 'Fe': '#E06633', 'Co': '#F090A0'
}
DEFAULT_COLOR = '#CCCCCC'

class VisualizationCallback(BaseCallback):
    """
    Advanced Visualization Callback for HEA Surfaces.
    Generates:
    1. Static PNG (Top/Side views)
    2. Structure CIF file
    3. Interactive HTML (Plotly 3D)
    """
    def __init__(self, save_freq: int, save_dir: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 获取环境实例
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'atoms') and env.atoms is not None:
                atoms = env.atoms
                step = self.num_timesteps
                energy = env.current_energy if hasattr(env, 'current_energy') else 0.0
                
                # --- 1. 保存 CIF 文件 ---
                cif_path = os.path.join(self.save_dir, f"step_{step:06d}.cif")
                write(cif_path, atoms)
                
                # --- 2. 生成交互式 HTML (Plotly) ---
                if ADVANCED_VIS:
                    html_path = os.path.join(self.save_dir, f"step_{step:06d}.html")
                    title = f"Step {step} | E={energy:.3f} eV | {atoms.get_chemical_formula()}"
                    plot_structure_plotly(atoms, title, html_path)

                # --- 3. 生成静态 PNG (Matplotlib) ---
                fig = plt.figure(figsize=(12, 6), dpi=150)
                gs = GridSpec(1, 2, width_ratios=[1, 1.2])
                
                # Top View
                ax_top = fig.add_subplot(gs[0])
                plot_atoms(atoms, ax_top, radii=0.8, rotation='-90x', show_unit_cell=2)
                ax_top.set_title("Top View", fontsize=12, fontweight='bold')
                ax_top.axis('off')

                # Side View
                ax_side = fig.add_subplot(gs[1])
                plot_atoms(atoms, ax_side, radii=0.8, rotation='-90x,90y', show_unit_cell=0)
                
                # Active Region Boundary
                positions = atoms.get_positions()
                z_max = positions[:, 2].max()
                layer_spacing = 2.2
                active_depth = env.n_active_layers * layer_spacing
                active_boundary = z_max - active_depth + (layer_spacing * 0.5)
                ax_side.axvline(x=active_boundary, color='red', linestyle='--', alpha=0.5)
                
                ax_side.set_title(f"Side View (Active: {env.n_active_layers})", fontsize=12, fontweight='bold')
                ax_side.axis('off')

                # Info Text
                info_text = f"Step: {step}\nEnergy: {energy:.3f} eV\n{atoms.get_chemical_formula()}"
                fig.text(0.02, 0.95, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

                # Legend
                unique_elements = sorted(list(set(atoms.get_chemical_symbols())))
                patches = [plt.Circle((0,0), color=ELEMENT_COLORS.get(e, DEFAULT_COLOR), label=e) for e in unique_elements]
                fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.98, 0.95))

                plt.tight_layout()
                png_path = os.path.join(self.save_dir, f"step_{step:06d}.png")
                plt.savefig(png_path, bbox_inches='tight')
                plt.close(fig)
                
        return True