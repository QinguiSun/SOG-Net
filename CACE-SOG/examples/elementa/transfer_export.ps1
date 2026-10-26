<#
Generate SHA256SUMS, then optionally upload an export directory with WinSCP SFTP.
Create and verify the stored SFTP site in the WinSCP GUI first. No password or
host-key bypass is written into this script. Remote directory must already exist.
#>
[CmdletBinding()]
param(
    [string]$LocalExport = 'C:\Users\qingu\Documents\elementa_cace_sog_windows\export',
    [ValidatePattern('^[^"\r\n]+$')]
    [string]$RemoteExport = '/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export',
    [ValidatePattern('^[^"\r\n]*$')]
    [string]$StoredSession = '',
    [string]$WinScpCom = '',
    [switch]$ChecksumOnly
)
$ErrorActionPreference = 'Stop'
$ElementaLocal = (Resolve-Path -LiteralPath $LocalExport).Path.TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $ElementaLocal 'manifest.jsonl') -PathType Leaf)) {
    throw 'The export directory must contain manifest.jsonl.'
}
foreach ($ElementaStatusName in @('summary.json', 'run_status.json')) {
    $ElementaStatus = Get-Content -LiteralPath (Join-Path $ElementaLocal $ElementaStatusName) -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($ElementaStatus.status -ne 'complete') { throw "$ElementaStatusName must report complete before upload." }
}
$ElementaChecksums = Join-Path $ElementaLocal 'SHA256SUMS'
$ElementaChecksumTemp = Join-Path $ElementaLocal 'SHA256SUMS.tmp'
$ElementaWriter = [System.IO.StreamWriter]::new($ElementaChecksumTemp, $false, [System.Text.UTF8Encoding]::new($false))
$ElementaWriter.NewLine = "`n"
try {
    Get-ChildItem -LiteralPath $ElementaLocal -Recurse -File | Where-Object {
        $_.FullName -ne $ElementaChecksums -and $_.FullName -ne $ElementaChecksumTemp
    } | Sort-Object FullName | ForEach-Object {
        $ElementaRelative = $_.FullName.Substring($ElementaLocal.Length + 1).Replace('\', '/')
        if ($ElementaRelative.Contains("`n") -or $ElementaRelative.Contains("`r")) { throw 'Newlines in export filenames are unsupported.' }
        Write-Host "Hashing $ElementaRelative"
        $ElementaHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $ElementaWriter.WriteLine("$ElementaHash  $ElementaRelative")
    }
} finally {
    $ElementaWriter.Dispose()
}
Move-Item -LiteralPath $ElementaChecksumTemp -Destination $ElementaChecksums -Force
Write-Host "Created $ElementaChecksums"
if ($ChecksumOnly) { return }
if ([string]::IsNullOrWhiteSpace($StoredSession)) {
    throw 'Specify -StoredSession with a verified SFTP site saved in WinSCP, or use -ChecksumOnly.'
}
if ([string]::IsNullOrWhiteSpace($WinScpCom)) {
    $ElementaCandidates = foreach ($ElementaProgramRoot in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if (-not [string]::IsNullOrWhiteSpace($ElementaProgramRoot)) {
            Join-Path $ElementaProgramRoot 'WinSCP\WinSCP.com'
        }
    }
    $WinScpCom = $ElementaCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($WinScpCom) -or -not (Test-Path -LiteralPath $WinScpCom -PathType Leaf)) {
    throw 'WinSCP.com was not found. Install WinSCP or provide -WinScpCom with its actual location.'
}
if ($ElementaLocal.Contains('"') -or $ElementaLocal.Contains("`n") -or $ElementaLocal.Contains("`r")) {
    throw 'Quotes or newlines in the local export path are unsupported.'
}
$ElementaScript = Join-Path ([System.IO.Path]::GetTempPath()) ('elementa-winscp-' + [guid]::NewGuid().ToString('N') + '.txt')
$ElementaCommands = @(
    'option batch abort',
    'option confirm off',
    ('open "' + $StoredSession + '"'),
    ('synchronize remote -criteria=size -transfer=binary -resumesupport=on "' + $ElementaLocal + '" "' + $RemoteExport + '"'),
    'exit'
)
[System.IO.File]::WriteAllLines($ElementaScript, $ElementaCommands, [System.Text.UTF8Encoding]::new($true))
try {
    & $WinScpCom ("/script=" + $ElementaScript)
    if ($LASTEXITCODE -ne 0) { throw "WinSCP upload failed with exit code $LASTEXITCODE. Rerun with the same, unchanged source directory to resume." }
} finally {
    Remove-Item -LiteralPath $ElementaScript -ErrorAction SilentlyContinue
}
Write-Host 'Upload command completed. On SCNet, run sha256sum -c SHA256SUMS inside the remote export directory.'
