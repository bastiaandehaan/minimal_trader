Write-Host "================================"
Write-Host "MINIMAL TRADER TEST SUITE"
Write-Host "================================"

Write-Host "`n[1/3] Unit + property tests..."
python -m pytest -v

Write-Host "`n[2/3] RSI tests..."
python -m pytest tests/test_rsi_strategy.py -v

Write-Host "`n[3/3] Smoke integration..."
python -m pytest -k integration -v

Write-Host "`nALL TESTS COMPLETED"
Write-Host "================================"
