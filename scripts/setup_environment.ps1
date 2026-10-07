<#
Create the exact course environment on Windows with Python 3.11.8.

The course pins mlflow==2.14.1 and pyarrow==16.1.0, but the former declares
pyarrow<16. A normal resolver therefore cannot install requirements.txt. This
script installs all non-conflicting pins, lets MLflow install its dependencies,
and finally applies the required PyArrow 16.1.0 wheel without dependency
resolution. It intentionally retains every top-level version specified in the
manual.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$requirementsPath = Join-Path $repositoryRoot 'requirements.txt'

if (-not (Test-Path -LiteralPath $venvPython)) {
    & py -3.11 -m venv (Join-Path $repositoryRoot '.venv')
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the Python 3.11 virtual environment.'
    }
}

$pythonVersion = & $venvPython -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to query the virtual-environment Python version.'
}
if ($pythonVersion -ne '3.11.8') {
    throw "Expected Python 3.11.8 in .venv, found $pythonVersion. Recreate .venv with py -3.11."
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to upgrade pip.'
}

$nonConflictingPins = Get-Content -LiteralPath $requirementsPath |
    Where-Object { $_ -and $_ -notmatch '^(mlflow|pyarrow)==' }
& $venvPython -m pip install --progress-bar off @nonConflictingPins
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the non-conflicting required packages.'
}

# Install MLflow and all of its ordinary dependencies while it may select its
# declared compatible PyArrow dependency. The next command applies the manual's
# explicit PyArrow 16.1.0 pin over that transitive dependency.
& $venvPython -m pip install --progress-bar off 'mlflow==2.14.1'
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install MLflow and its dependencies.'
}
& $venvPython -m pip install --progress-bar off --no-deps --force-reinstall 'pyarrow==16.1.0'
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to apply the required PyArrow pin.'
}

& $venvPython (Join-Path $repositoryRoot 'lab01\src\print_versions.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Environment version validation failed.'
}
