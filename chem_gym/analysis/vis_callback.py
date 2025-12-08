import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

class VisualizationCallback(BaseCallback):
    """
    自定义回调函数：每隔一定步数保存当前环境的原子结构图。
    """
    def __init__(self, save_freq: int, save_dir: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 检查是否达到保存频率
        if self.n_calls % self.save_freq == 0:
            # 获取第一个环境实例 (DummyVecEnv)
            env = self.training_env.envs[0]
            
            # 确保环境中有 atoms 对象
            if hasattr(env, 'atoms') and env.atoms is not None:
                # 使用 ASE 绘图
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_atoms(env.atoms, ax, show_unit_cell=0, rotation='-90x')
                
                # 添加标题信息
                energy = env.current_energy if hasattr(env, 'current_energy') else 0.0
                ax.set_title(f"Step {self.num_timesteps} | E={energy:.3f} eV")
                ax.axis('off')
                
                # 保存图片
                save_path = os.path.join(self.save_dir, f"step_{self.num_timesteps}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                
        return True