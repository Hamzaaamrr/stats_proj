# run_all.R - re-run every analysis script after recording new data.
# Open analysis.Rproj, open this file, click "Source".
# All charts go to output/figures, all tables to output/tables.

scripts <- c("02_session_report.R",   # Student B - what happened in each trip
             "03_risk_score.R",       # Student C - driver risk score
             "04_stat_tests.R",       # Student D - are the differences real?
             "05_accuracy.R")         # Student E - can the system be trusted?

for (s in scripts) {
  cat("\n==================== ", s, " ====================\n")
  source(s)
}
cat("\nAll done. Dashboard: open app.R and click 'Run App'.\n")
