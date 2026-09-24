import sys
from pathlib import Path

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title.upper()}")
    print("=" * 60)

def check_directories(base_path: Path):
    print("\n[+] Checking Project Directory Architecture...")
    required_dirs = ["data", "models", "notebooks", "src"]
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        status = "OK" if dir_path.exists() and dir_path.is_dir() else "MISSING"
        print(f"  |-- Directory {dir_name:<12} : [{status}] -> {dir_path.name}/")
        if status == "MISSING":
            all_ok = False
            
    return all_ok

def check_dependencies():
    print("\n[+] Checking Installed Python Libraries...")
    packages = [
        ("Python Engine", None),
        ("pandas", "pandas"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("pyarrow", "pyarrow"),
        ("joblib", "joblib"),
        ("kaggle", "kaggle"),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        if import_name is None:
            v = sys.version.split()[0]
            print(f"  |-- {pkg_name:<15} : [OK] (Version: {v})")
            continue
            
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  |-- {pkg_name:<15} : [OK] (Version: {ver})")
        except ImportError:
            print(f"  |-- {pkg_name:<15} : [FAILED] (Not Installed)")
            all_ok = False
            
    return all_ok

def main():
    project_root = Path(__file__).resolve().parent
    
    print_header("NIDS Software System - Health Check (Step 1)")
    print(f"Project Root Path: {project_root}")
    
    dirs_ok = check_directories(project_root)
    pkgs_ok = check_dependencies()
    
    print_header("Step 1 Final Verification Result")
    if dirs_ok and pkgs_ok:
        print("  [SUCCESS] All components verified. Ready for Step 2.")
    else:
        print("  [WARNING] Missing requirements. Check failed items above.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()