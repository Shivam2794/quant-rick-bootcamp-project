$python = "C:\Users\Shivam Patel\.gemini\antigravity\brain\7b03663a-d01b-4302-8959-0a511c484299\.venv\Scripts\python.exe"
& $python data_ingestion.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python meta_regime_filter.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python momentum_prefilter.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python raam_scorer.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python rsi_cross_sectional_engine.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python position_sizer.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python backtest_engine.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python bt2_video1_carver.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python bt2_video2_donchian.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python bt2_video3_macro_canaries.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
