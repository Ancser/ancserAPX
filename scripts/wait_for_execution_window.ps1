param(
    [datetime]$NowNy = [datetime]::MinValue,
    [switch]$NoSleep
)

$ErrorActionPreference = "Stop"
$NewYorkZone = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
if ($NowNy -eq [datetime]::MinValue) {
    $NowNy = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $NewYorkZone)
}

$TargetNy = $NowNy.Date.AddHours(9).AddMinutes(35)
$WaitSeconds = [int][Math]::Ceiling(($TargetNy - $NowNy).TotalSeconds)

# Backward compatibility for an already-installed privileged 09:25 ET task.
# Never turn a missed/late task into an hours-long sleeper: only bridge launches
# that are already within 15 minutes of today's controlled execution time.
if ($WaitSeconds -gt 0 -and $WaitSeconds -le 900) {
    Write-Output ("Waiting {0}s for 09:35 America/New_York execution window." -f $WaitSeconds)
    if (-not $NoSleep) {
        Start-Sleep -Seconds $WaitSeconds
    }
} else {
    $WaitSeconds = 0
    Write-Output "No execution-window wait required."
}

Write-Output ("Execution-window wait_seconds={0}." -f $WaitSeconds)
