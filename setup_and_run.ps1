Write-Host "Verificando/Criando ambiente virtual (venv)..."

# Se não existir a pasta venv, criar
if (-not (Test-Path "$PSScriptRoot\venv")) {
    Write-Host "Criando nova venv..."
    python -m venv "$PSScriptRoot\venv"
}

Write-Host "Ativando venv..."
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

Write-Host "Instalando dependências..."
pip install -r "$PSScriptRoot\requirements.txt"

Write-Host "Iniciando a aplicação Streamlit..."
streamlit run "$PSScriptRoot\app.py"
