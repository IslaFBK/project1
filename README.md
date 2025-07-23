# Statement
**This repository contain both original and non-original code. Its main purpose is to facilitate my personal learning. So, please ignore this repository and consider it as a private one.**
# Usage
Run(.py) compute_parallel > draw_auto $\ge$ state_parallel_auto > interactive_states.
Means: compute raw data > draw jump step, coactivity distribution, generate pattern vedio $\ge$ statisticize states > build an interactive GUI for check states distribution in 3D phase space.

The folder `analysis`, `connection`, and `model_file` are from the beta version of [Distributed_and_dynamical_communication from Shencong-Ni](https://github.com/BrainDynamicsUSYD/Distributed_and_dynamical_communication)

# Setup
## Anaconda Prompt:
```python
conda create --prefix "X:\path\.venv" python=3.11.7
conda activate "X:\path\.venv"
```
## VsCode:
In the vary folder:
```
python.exe -m pip install -r packages.txt
python.exe -m pip install -r requirements.txt
```
## Windows:
### Add the three folders to path:
1. Open system property > advanced > environment variable;
2. Find or establish "PYTHONPATH" in system variable;
3. Add "X:\path\analysis;X:\path\connection;X:\path\model_file" to value, multiple path separated by ";";

### Install Microsoft C++ Build Tools:
1. Open Microsoft C++ Build Tools page([Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/))
2. Download and run installer(`Visual Studio Installer`), select following option (or just select `Desktop development with C++`):
   1. `C++ Build Tools work` load;
   2. (`Windows 10 SDK` and `MSVC v142/v143`).
3. Restart PC after installed above tools.
   
### Install `ffmpeg` 
See (https://blog.csdn.net/qq_45956730/article/details/125272407).
### interactive phase diagram:
1. requirements: ipywidgets
2. Jupyter support:
   - Run in **Jupyter Notebook** need enable `widgets`:
   ```
   jupyter nbextension enable --py widgetsnbextension
   ```
### LaTeX install:
See (https://zhuanlan.zhihu.com/p/166523064).

# Remote coding
## Update `requirements.txt` and `packages.txt`
```
pip freeze > requirements.txt
pip list --format=freeze > packages.txt
```
## prox setting
The command to use:
```bash
git config --global http.proxy http://proxyuser:proxypwd@proxy.server.com:8080
```
- change `proxyuser` to your proxy user
- change `proxypwd` to your proxy password
- change `proxy.server.com` to the URL of your proxy server

For example:
```bash
git config --global http.proxy http://127.0.0.1:4780
git config --global https.proxy https://127.0.0.1:4780
# may be unnecessary
git config --global http.proxy socks5://127.0.0.1:4780
```
## .gitconfig
Here's a successful `.gitconfig`:
```
# This is Git's per-user configuration file.
[user]
	name = your_name
	email = your_email@example.example
# Please adapt and uncomment the following lines:
#	name = unknown
#	email = work@DESKTOP-2U2AG9K.(none)
[http]
	proxy = http://127.0.0.1:4780
[https]
	proxy = https://127.0.0.1:4780
```

## SSH from PC and iPad (Code app) setting
SSH in iPad need no extra setting, following the tutorial of Cpolar will be enough.
**If something wrong, try to run as administrator while operating OS in the server.**  
### Between intranet Windows
[Tutorial](https://zhuanlan.zhihu.com/p/339012691)
`C:\User\.ssh\config` in PC:
```
Host server_user
    HostName 10.181.252.21(Intranet IP of your server)
    User server_user
```
### sshd_config in server  
Not clear now, but following sshd_config works:
```sshd_config
Port 22
ListenAddress 0.0.0.0
PasswordAuthentication yes
PermitRootLogin yes
Subsystem sftp sftp-server.exe

Match User work
    ChrootDirectory X:\your_path
    ForceCommand none
    PermitTTY yes
    X11Forwarding no
    AllowTcpForwarding no
    PermitTunnel no
    Subsystem sftp internal-sftp -u 0002
```
If server's sshd_config has been modified by your fellow, possibly cause folder empty, or even cannot connect. As I know, sshd_config is all you need about SSH.  
### NTFS authority
- User `your_account` should **full control** `X:\your_path` and its sub-content
- System account `sshd` need **Read/Execute** permission

**Configuration**
```powershell
# Grant the your_account user full control privileges
icacls "X:\your_path" /grant "your_account:(OI)(CI)(F)" /T

# Grant the sshd service account read access permission
icacls "X:\your_path" /grant "NT SERVICE\sshd:(RX)"
```
**Verify permissions**
```powershell
icacls "X:\your_path"
```
**Output should contain:**
```
your_account:(OI)(CI)(F)
NT SERVICE\sshd:(RX)
```
If something wrong, try rigth-click your path in folder > property > Security, manually add accounts and permissions.
### Check Chroot menu structure
Chroot specially require:
- `X:\your_path` has to be a **genuine directory** (non-symlink/mount point)
- The permissions of the directory and all its parent directories (up to the root directory)
   - `Administrators` and `SYSTEM` must have full control
   - Other users need at least the permission to traverse

**Repair command**
```powershell
# set permission of parent directories
icacls "F:\" /grant "your_account:(RX)"
icacls "F:\" /grant "Administrators:(OI)(CI)(F)"
icacls "F:\" /grant "SYSTEM:(OI)(CI)(F)"
```
If something wrong, try rigth-click your path in folder > property > Security, manually add accounts and permissions.
### Cpolar (Intranet penetration)
[Tutorial](https://www.bilibili.com/video/BV1Vj411W7W9)  
[Cpolar official website](https://www.cpolar.com/)  
[Cpolar local host](http://localhost:9200/)  
`C:\Users\.cpolar\cpolar.yml` in server:
```
authtoken: CpolarAuthtokenExample
tunnels:
  code1:
    id: your-id-example
    proto: tcp
    addr: "22"
    inspect: "false"
    bind_tls: both
    region: cn_top
    disable_keep_alives: "false"
    redirect_https: "false"
    start_type: enable
email: your_email@example.example
```