# Script de prueba de balanceo de carga en Kubernetes
$currentDir = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$shFile = Join-Path $currentDir "send_traffic.sh"
$jsonPayload = '{\"MedInc\":3.5,\"HouseAge\":25.0,\"AveRooms\":5.2,\"AveBedrms\":1.05,\"Population\":1200.0,\"AveOccup\":3.0,\"Latitude\":37.88,\"Longitude\":-122.23}'
$script = "for i in 1 2 3 4 5 6; do`n  curl -s -X POST -H ""Content-Type: application/json"" -d ""$jsonPayload"" http://housing-ml-service/predict`n  echo """"`ndone`n"
[System.IO.File]::WriteAllText($shFile, $script)
Get-Content -Raw $shFile | kubectl run client-test --image=curlimages/curl --restart=Never --rm -i --quiet -- sh
Remove-Item $shFile -Force -ErrorAction SilentlyContinue
