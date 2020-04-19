from pathlib import Path
import sys
import os


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    for x in (folder / "meshes").iterdir():
        tgt_dir = folder / x.name.split(".")[0]
        tgt_dir.mkdir(exist_ok=True)
        os.rename(x, tgt_dir / "model.obj")
        os.rename(folder / "points" / (x.name.split(".")[0] + ".npz"), tgt_dir / "points.npz")
