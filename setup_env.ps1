Write-Host "Creating virtual environment .venv..."
$python = "python"
try {
    $pyver = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
} catch {
    Write-Error "Python not found on PATH. Use the full path to python or run from the folder with python available."
    exit 1
}

if (-not (Test-Path -Path .venv)) {
    & $python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create virtualenv"; exit 1 }
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing requirements from requirements.txt..."
try {
    python -m pip install -r requirements.txt
} catch {
    Write-Warning "Failed to install one or more packages from requirements.txt. Trying a PyTorch CPU fallback..."
    Write-Host "Attempting CPU-PyTorch install from official index..."
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Automatic installation of torch failed. This is commonly due to missing wheel for your Python version (e.g., Python 3.13)."
        Write-Host "Recommended actions:"
        Write-Host "  1) Install Python 3.11 (https://www.python.org/downloads/) and re-run this script."
        Write-Host "  2) Or install torch manually with a wheel compatible with your Python version."
        exit 1
    }
}

Write-Host "Setup complete. To run the app in this shell, ensure the venv is activated and run:\npython ImageCaptioning.py"
