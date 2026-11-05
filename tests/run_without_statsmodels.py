"""用「擋掉 statsmodels、scipy、scikit-learn」的方式執行 run_model.py，模擬 claude.ai 沒有這些套件的環境。

用法：python tests/run_without_statsmodels.py [--no-matplotlib] <run_model.py 的參數…>
只要 run_model.py 或 catfit.py 想 import 這些套件，就會直接出錯。
"""
import os
import runpy
import sys

BLOCK = ["statsmodels", "scipy", "sklearn", "patsy"]
args = sys.argv[1:]
if args and args[0] == "--no-matplotlib":
    BLOCK.append("matplotlib")
    args = args[1:]

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError(f"這個測試環境擋掉了 {name}（模擬 claude.ai 沒有這個套件）")
        return None

sys.meta_path.insert(0, Blocker())
here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(os.path.dirname(here), "scripts", "run_model.py")
sys.argv = [script] + args
runpy.run_path(script, run_name="__main__")
