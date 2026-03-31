import subprocess
import os
import sys

def check_nvidia_gpu():
    """Detect if an NVIDIA GPU is present on the system."""
    try:
        # Standard way to check via driver
        res = subprocess.run(["nvidia-smi"], capture_output=True)
        if res.returncode == 0: return True
    except: pass
    
    try:
        # Alternate way via Windows management
        res = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], capture_output=True, text=True)
        if "NVIDIA" in res.stdout.upper(): return True
    except: pass
    return False

def check_apple_silicon():
    """Detect if the system is an Apple Silicon Mac."""
    import platform
    return sys.platform == "darwin" and platform.machine() == "arm64"

def check_torch_cuda():
    """Check if the currently installed PyTorch already has CUDA capability."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def install():
    has_gpu = check_nvidia_gpu()
    is_apple_silicon = check_apple_silicon()
    already_cuda = check_torch_cuda()
    python = sys.executable
    
    print("\n" + "="*50)
    print("  CV-COUNT SMART INSTALLER -- SCANNING HARDWARE")
    print("="*50)
    
    if is_apple_silicon:
        print("\n[DETECTED] Apple Silicon Mac Found! (M1/M2/M3/M4)")
        print("[ACTION]   Ready for Apple Metal (MPS) acceleration.")
        cmd = f"{python} -m pip install torch torchvision torchaudio" # Standard torch has MPS
    elif has_gpu and already_cuda:
        print("\n[SUCCESS] NVIDIA GPU Detected and PyTorch is already CUDA-optimized!")
        print("[ACTION]  Skipping 2.6GB download. Using existing environment.")
        cmd = None
    elif has_gpu:
        print("\n[DETECTED] NVIDIA GPU Found! (Perfect for RTX 3050, etc.)")
        print("="*50)
        print("  IMPORTANT: High-Performance GPU Support")
        print("  The CUDA-enabled PyTorch is ~2.6GB.")
        print("="*50)
        
        choice = input("\nDo you want to download the 2.6GB GPU-optimized version? (Y/N) [Y]: ").strip().lower()
        
        if choice in ('', 'y', 'yes'):
            print("\n[ACTION]   Optimizing environment for CUDA Acceleration (2.6GB download)...")
            cmd = f"{python} -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 --force-reinstall"
        else:
            print("\n[ACTION]   Skipping large download. Installing standard CPU version...")
            cmd = f"{python} -m pip install torch torchvision torchaudio"
    else:
        print("\n[DETECTED] No NVIDIA GPU found (or drivers not installed).")
        print("[ACTION]   Installing standard CPU-only environment...")
        cmd = f"{python} -m pip install torch torchvision torchaudio"

    if cmd:
        print("\nRunning: " + cmd)
        subprocess.run(cmd, shell=True)
    
    print("\n[ACTION]   Installing remaining dependencies (ultralytics, opencv, lap)...")
    subprocess.run(f"{python} -m pip install -r requirements.txt", shell=True)
    
    print("\n" + "="*50)
    print("  SETUP COMPLETE!")
    print("="*50)
    
    choice = input("\nWould you like to launch CV-Count now? (Y/N) [Y]: ").strip().lower()
    if choice in ('', 'y', 'yes'):
        print("\n[ACTION]   Launching the application...")
        # Use Popen so the setup window can close while the app stays open
        try:
            # 0x00000010 is CREATE_NEW_CONSOLE on Windows
            subprocess.Popen(["cmd.exe", "/c", "03. Start CV Count App (Run after 02).bat"], 
                             creationflags=0x00000010)
        except:
            # Fallback to direct python call if bat fails
            subprocess.Popen([python, "counter.py"], 
                             creationflags=0x00000010)

if __name__ == "__main__":
    install()
