import requests
import os
from tqdm import tqdm

# 1. 切换到镜像站地址
BASE_URL = "https://hf-mirror.com/facebook/UMA/resolve/main"


# 官方要求的参考文件
files = {
    f"{BASE_URL}/references/form_elem_refs.yaml": "checkpoints/references/form_elem_refs.yaml",
    f"{BASE_URL}/references/iso_atom_elem_refs.yaml": "checkpoints/references/iso_atom_elem_refs.yaml"
}

def download_with_mirror(url, dest):
    print(f"正在下载: {url}")
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    
    if response.status_code == 200:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 成功保存至: {dest}")
    else:
        print(f"❌ 下载失败: {response.status_code}")

if __name__ == "__main__":
    for url, path in files.items():
        download_with_mirror(url, path)