<#
CPU-only preparation: complete training index, then seal a portable bundle.
Requires only NumPy and ASE. It does not import the model stack, use CUDA,
connect to SCNet, or submit any jobs. Keep source export files unchanged.
#>
[CmdletBinding()]
param(
    [string]$PythonExe = 'python',
    [string]$ProjectDir = '',
    [string]$ExportDir = 'C:\Users\qingu\Documents\elementa_cace_sog_windows\export',
    [string]$WorkDir = 'C:\Users\qingu\Documents\elementa_cace_sog_windows\training',
    [string]$Database = '',
    [string]$Bundle = '',
    [ValidateSet('sog_candidates', 'lattice_auxiliary', 'pbe_lowdim_auxiliary', 'spin_review')]
    [string[]]$Channels = @('sog_candidates', 'lattice_auxiliary', 'pbe_lowdim_auxiliary'),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = Join-Path $PSScriptRoot '..\..'
}
$ElementaProject = (Resolve-Path -LiteralPath $ProjectDir).Path
$ElementaExport = (Resolve-Path -LiteralPath $ExportDir).Path
$ElementaWork = [System.IO.Path]::GetFullPath($WorkDir)
if (-not (Test-Path -LiteralPath (Join-Path $ElementaProject 'cace_sog_training\__main__.py') -PathType Leaf)) {
    throw 'ProjectDir must be the adapted SOG-Net\CACE-SOG source directory.'
}
if (-not (Test-Path -LiteralPath (Join-Path $ElementaExport 'manifest.jsonl') -PathType Leaf)) {
    throw 'ExportDir must contain the completed export manifest.jsonl.'
}
if ($Channels.Count -eq 0 -or @($Channels | Select-Object -Unique).Count -ne $Channels.Count) {
    throw 'Choose distinct channels; the default retains all frames from the first three channels.'
}
if ([string]::IsNullOrWhiteSpace($Database)) { $Database = Join-Path $ElementaWork 'elementa.sqlite3' }
if ([string]::IsNullOrWhiteSpace($Bundle)) { $Bundle = Join-Path $ElementaWork 'bundle' }
$ElementaDatabase = [System.IO.Path]::GetFullPath($Database)
$ElementaBundle = [System.IO.Path]::GetFullPath($Bundle)
foreach ($ElementaLocalPath in @($ElementaWork, $ElementaDatabase, $ElementaBundle)) {
    if ($ElementaLocalPath.StartsWith('\\')) { throw 'Preparation work/index/bundle must be on a spacious local Windows drive, not a UNC share.' }
    $ElementaDriveRoot = [System.IO.Path]::GetPathRoot($ElementaLocalPath)
    $ElementaDrive = [System.IO.DriveInfo]::new($ElementaDriveRoot)
    if ($ElementaDrive.DriveType -eq [System.IO.DriveType]::Network) { throw 'A mapped network drive is not a local SQLite workspace.' }
}
if (Test-Path -LiteralPath $ElementaBundle) {
    throw 'Bundle must be a NEW directory. Keep a complete bundle unchanged, or use -Bundle with a new local directory after an interrupted seal.'
}
if ((Test-Path -LiteralPath $ElementaDatabase) -and -not $Resume) {
    throw 'The training index exists. To continue the same preparation, use -Resume with unchanged source paths and channels.'
}
New-Item -ItemType Directory -Path $ElementaWork -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $ElementaDatabase) -Force | Out-Null
$ElementaTemp = Join-Path $ElementaWork '.sqlite_tmp'
New-Item -ItemType Directory -Path $ElementaTemp -Force | Out-Null
$ElementaEnvironmentNames = @('PYTHONPATH', 'PYTHONUTF8', 'PYTHONUNBUFFERED', 'SQLITE_TMPDIR', 'TEMP', 'TMP')
$ElementaPreviousEnvironment = @{}
foreach ($ElementaName in $ElementaEnvironmentNames) {
    $ElementaPreviousEnvironment[$ElementaName] = [System.Environment]::GetEnvironmentVariable($ElementaName, 'Process')
}
$ElementaPushedLocation = $false
try {
    $env:PYTHONPATH = $ElementaProject
    if (-not [string]::IsNullOrEmpty($ElementaPreviousEnvironment['PYTHONPATH'])) {
        $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $ElementaPreviousEnvironment['PYTHONPATH']
    }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:SQLITE_TMPDIR = $ElementaTemp
    $env:TEMP = $ElementaTemp
    $env:TMP = $ElementaTemp
    Push-Location -LiteralPath $ElementaProject
    $ElementaPushedLocation = $true
    & $PythonExe -X utf8 -u -c 'import sys, numpy, ase; print(sys.executable); print(numpy.__version__); print(ase.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'The selected Python must have NumPy and ASE installed.' }
    $ElementaIndexArguments = @('-X', 'utf8', '-u', '-m', 'cace_sog_training', 'index', '--export-dir', $ElementaExport, '--database', $ElementaDatabase, '--channels') + $Channels
    if ($Resume) { $ElementaIndexArguments += '--resume' }
    & $PythonExe @ElementaIndexArguments
    if ($LASTEXITCODE -ne 0) { throw 'Index preparation did not complete. Preserve the index and retry with -Resume after resolving the reported input error.' }
    & $PythonExe -X utf8 -u -m cace_sog_training seal --database $ElementaDatabase --bundle $ElementaBundle
    if ($LASTEXITCODE -ne 0) { throw 'Sealing did not complete. Keep the complete training index; use a NEW -Bundle path for the next seal attempt.' }
    Write-Host 'Preparation complete. Upload the sealed bundle and the export files listed in bundle.json once, using binary transfer.'
    Write-Host "Training index: $ElementaDatabase"
    Write-Host "Sealed bundle: $ElementaBundle"
} finally {
    if ($ElementaPushedLocation) { Pop-Location }
    foreach ($ElementaName in $ElementaEnvironmentNames) {
        [System.Environment]::SetEnvironmentVariable($ElementaName, $ElementaPreviousEnvironment[$ElementaName], 'Process')
    }
}
