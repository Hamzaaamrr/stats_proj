# 05_accuracy.R - can the system be trusted? (Student E)
# Uses test-mode sessions, where a second person pressed keys for what the
# driver was REALLY doing (True_Label). We compare that with what the app said.
source("01_load_data.R")

test <- samples %>%
  filter(!is.na(True_Label)) %>%
  arrange(Session_ID, Time_s)
cat("\nLabeled seconds:", nrow(test), "from sessions", paste(unique(test$Session_ID), collapse = ", "), "\n")


# ---- 1. Overall accuracy: strict and with +/- 2 seconds tolerance -------------
# The app waits 1.5-2 s on purpose before labeling, and the person pressing the
# key reacts late too. So we also count a second as correct if the app's label
# matches the true label up to 2 seconds earlier or later.
same <- function(a, b) coalesce(a == b, FALSE)

test <- test %>%
  group_by(Session_ID) %>%
  mutate(Correct_Tolerant = same(Behavior, True_Label) |
           same(Behavior, lag(True_Label, 1))  | same(Behavior, lag(True_Label, 2)) |
           same(Behavior, lead(True_Label, 1)) | same(Behavior, lead(True_Label, 2))) %>%
  ungroup()

overall <- data.frame(
  Method   = c("Strict (same second)", "Tolerant (+/- 2 s)"),
  Accuracy = round(100 * c(mean(test$Correct), mean(test$Correct_Tolerant)), 1)
)
print(overall)
write_csv(overall, "output/tables/05_overall_accuracy.csv")


# ---- 2. Confusion matrix ------------------------------------------------------
# Rows = what really happened, columns = what the app said.
confusion <- table(True = test$True_Label, Predicted = test$Behavior)
print(confusion)
write_csv(as.data.frame(confusion), "output/tables/05_confusion_matrix.csv")

# For the chart: each row as % (so every true behavior's row adds up to 100%)
confusion_df <- as.data.frame(confusion) %>%
  group_by(True) %>%
  mutate(Row_Total = sum(Freq)) %>%
  ungroup() %>%
  filter(Row_Total > 0) %>%                         # skip behaviors never acted out
  mutate(Row_Percent = round(100 * Freq / Row_Total))

p1 <- ggplot(confusion_df, aes(x = Predicted, y = True, fill = Row_Percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = paste0(Row_Percent, "%\n(", Freq, ")")), size = 3.2) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  scale_y_discrete(limits = rev) +
  labs(title = "Confusion matrix: what really happened vs what the app said",
       subtitle = "Each row adds up to 100%. The diagonal = correct. (n) = seconds",
       x = "App said", y = "Really happened", fill = "% of row") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p1)
ggsave("output/figures/05_confusion_matrix.png", p1, width = 8, height = 6, bg = "white")


# ---- 3. Precision and recall per behavior -------------------------------------
#   Precision: when the app SAYS "Phone", how often is it right?
#   Recall:    of all REAL phone use, how much did the app catch?
#   F1:        one number combining both (higher = better)
per_behavior <- data.frame(Behavior = behaviors) %>%
  rowwise() %>%
  mutate(Real_Seconds      = sum(test$True_Label == Behavior),
         Predicted_Seconds = sum(test$Behavior == Behavior),
         Correct_Seconds   = sum(test$True_Label == Behavior & test$Behavior == Behavior),
         Precision = round(100 * Correct_Seconds / Predicted_Seconds, 1),
         Recall    = round(100 * Correct_Seconds / Real_Seconds, 1),
         F1        = round(2 * Precision * Recall / (Precision + Recall), 1)) %>%
  ungroup()
print(per_behavior)
write_csv(per_behavior, "output/tables/05_precision_recall.csv")

pr_long <- per_behavior %>%
  filter(Real_Seconds > 0) %>%
  select(Behavior, Precision, Recall) %>%
  tidyr::pivot_longer(c(Precision, Recall), names_to = "Measure", values_to = "Percent") %>%
  mutate(Behavior = factor(Behavior, levels = behaviors),
         Label    = ifelse(is.na(Percent), "n/a", Percent),   # n/a = the app never said it
         Percent  = coalesce(Percent, 0))

p2 <- ggplot(pr_long, aes(x = Behavior, y = Percent, fill = Measure)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = Label),
            position = position_dodge(width = 0.7), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = c("Precision" = "steelblue", "Recall" = "darkorange")) +
  scale_y_continuous(limits = c(0, 105)) +
  labs(title = "Precision and recall per behavior",
       subtitle = "Precision: when the app says it, is it right?   Recall: how much of the real behavior did it catch?",
       x = NULL, y = "%", fill = NULL) +
  theme_minimal()
print(p2)
ggsave("output/figures/05_precision_recall.png", p2, width = 9, height = 5, bg = "white")


# ---- 4. Accuracy per session, before vs after the fixes, and by conditions ----
conditions <- sessions %>%
  select(Session_ID, Lighting, Camera_Position, Glasses, EAR_Baseline) %>%
  mutate(across(c(Lighting, Camera_Position, Glasses),
                ~ ifelse(is.na(.x), "unknown", as.character(.x))))

by_session <- test %>%
  group_by(Session_ID, Version) %>%
  summarise(Seconds = n(),
            Strict    = round(100 * mean(Correct), 1),
            Tolerant  = round(100 * mean(Correct_Tolerant), 1), .groups = "drop") %>%
  left_join(conditions, by = "Session_ID")
print(by_session)
write_csv(by_session, "output/tables/05_accuracy_by_session.csv")

p3 <- ggplot(by_session, aes(x = Session_ID, y = Tolerant, fill = Version)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(Tolerant, "%")), vjust = -0.4) +
  scale_fill_manual(values = c("before fix" = "grey60", "after fix" = "seagreen")) +
  scale_y_continuous(limits = c(0, 105)) +
  labs(title = "Accuracy per test session (+/- 2 s tolerance)",
       subtitle = "Grey = before the detection fixes, green = after",
       x = "Session", y = "Accuracy (%)", fill = NULL) +
  theme_minimal()
print(p3)
ggsave("output/figures/05_accuracy_by_session.png", p3, width = 8, height = 5, bg = "white")

by_version <- test %>%
  group_by(Version) %>%
  summarise(Seconds = n(),
            Strict   = round(100 * mean(Correct), 1),
            Tolerant = round(100 * mean(Correct_Tolerant), 1))
print(by_version)
write_csv(by_version, "output/tables/05_accuracy_by_version.csv")
if (n_distinct(test$Version) < 2)
  cat("\nNote: all test-mode sessions are '", unique(test$Version),
      "'. Record test-mode sessions with the new version to compare before vs after.\n", sep = "")

cat("\nDone. Charts are in output/figures, tables in output/tables.\n")
