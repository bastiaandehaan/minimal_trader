# Implementation Log

## Step 1: Backtest Smoke Test

**Date:** 2025-09-05  
**Who:** Bastiaan / ChatGPT  
**What:** Eerste validatie van strategie en engine via `python trader.py backtest --csv ...`  
**Why:** Verifiëren dat:
- De strategie initieert, signalen genereert, en trades uitvoert
- Er geen look-ahead bias is
- Output en logs goed geschreven worden
- `test_trader.py` draait zonder fouten

**Status:** ✅ Geslaagd  
- Testcases draaien met `pytest -v`
- Dummy data via `create_test_data()` werkt
- Backtest-metrics zijn gegenereerd en CSV wordt weggeschreven

**Next step:**  
→ Realistische data toevoegen in `data/` (bijv. GER40.cash_H1.csv)  
→ Run backtest op echte data  
→ Evalueren of logica/metrics kloppen

## Step 2: Architecture Refactoring to v2 Modules

**Date:** (auto)  
**What:** Split monolith into `strategy.py`, `backtest.py`, `data_feed.py`, `main.py` + tests  
**Why:** Separation of concerns, O(n) performance, testability

**Decisions:**
- Indicators computed once; signal per bar by index (no O(n²))
- BUY entries only; SELL used as exit trigger
- Time-exit fail-safe (200 bars)
- CSVFeed for input; LiveFeed later

**Status:** Implemented  
**Next:** Run tests + run backtest on real CSV (limit first)
