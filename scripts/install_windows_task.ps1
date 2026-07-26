param(
    [switch]$Refresh,
    [string]$TaskName = "ancserAPX Daily"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DailyBat = Join-Path $ProjectRoot "ancserAPX daily.bat"
if (-not (Test-Path -LiteralPath $DailyBat)) {
    throw "Daily launcher not found: $DailyBat"
}

# Windows uses this dynamic timezone ID for America/New_York. Converting the
# next 09:35 ET trigger to local time makes California 06:35 and lets any host
# timezone install without hand calculations.
$NewYorkZone = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$LocalZone = [TimeZoneInfo]::Local
$NowNy = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $NewYorkZone)
$TargetNy = $NowNy.Date.AddHours(9).AddMinutes(35)
if ($TargetNy -le $NowNy) {
    $TargetNy = $TargetNy.AddDays(1)
}
$TargetNyUnspecified = [DateTime]::SpecifyKind($TargetNy, [DateTimeKind]::Unspecified)
$TargetUtc = [TimeZoneInfo]::ConvertTimeToUtc($TargetNyUnspecified, $NewYorkZone)
$TargetLocal = [TimeZoneInfo]::ConvertTimeFromUtc($TargetUtc, $LocalZone)

$Cmd = Join-Path $env:SystemRoot "System32\cmd.exe"
$Action = New-ScheduledTaskAction `
    -Execute $Cmd `
    -Argument ('/d /c "' + $DailyBat + '"') `
    -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $TargetLocal
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Highest
$Task = New-ScheduledTask `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "ancserAPX staged rebalance at 09:35 America/New_York"

Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null

# Migrate the old manually-created task only when it points at this project's
# launcher. Never delete an unrelated task that merely shares the old name.
$Legacy = Get-ScheduledTask -TaskName "ancserFX" -ErrorAction SilentlyContinue
if ($Legacy) {
    $LegacyCommand = (($Legacy.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments)" }) -join " ")
    if ($LegacyCommand -like ("*" + $DailyBat + "*")) {
        Unregister-ScheduledTask -TaskName "ancserFX" -Confirm:$false
        Write-Output "Removed legacy duplicate task: ancserFX"
    }
}

$Mode = if ($Refresh) { "refreshed" } else { "installed" }
Write-Output ("Task '{0}' {1}: next local trigger {2:yyyy-MM-dd HH:mm:ss} ({3}); 09:35 America/New_York." -f `
    $TaskName, $Mode, $TargetLocal, $LocalZone.DisplayName)
