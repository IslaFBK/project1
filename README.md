# Statement
**This repository contain both original and non-original code. Its main purpose is to facilitate my personal learning. So, please ignore this repository and consider it as a private one.**
# Set steps
## Anaconda Prompt:
```python
conda create --prefix "X:\path\.venv" python=3.11.7
conda activate "X:\path\.venv"
pip install -r package.txt
```
## Windows:
### Add the three folder to path:
1. Open system property > advanced > environment variable;
2. Find or establish "PYTHONPATH" in system variable;
3. Add "X:\path\analysis;X:\path\connection;X:\path\connection\model_file" to value, multiple path separated by ";";

### Install Microsoft C++ Build Tools:
1. Open Microsoft C++ Build Tools page([Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/))
2. Download and run installer(`Visual Studio Installer`), select following option (or just select `Desktop development with C++`):
   1. `C++ Build Tools work` load;
   2. (`Windows 10 SDK` and `MSVC v142/v143`).
3. Restart PC after installed above tools.
   
### Install `ffmpeg` (see https://blog.csdn.net/qq_45956730/article/details/125272407);
### interactive phase diagram:
1. requirements: ipywidgets
2. Jupyter support:
   - Run in **Jupyter Notebook** need enable `widgets`:
   ```
   jupyter nbextension enable --py widgetsnbextension
   ```
### LaTeX install:
See (https://zhuanlan.zhihu.com/p/166523064)

## gitignore:

···python
*.pyc
*.png
*.jpg
*.svg
*.pdf
*.mp4
*.avi
*.zip
*.tar
*.mat
*.file
spkdata*
*.m~
*.py~
data*
data/
ijwd*
cache/
cache*/
__pycache__/
results/
analysis/load_data_dict.pyc
.svn/
.venv/
···

## ssh and code app setting
coming soon