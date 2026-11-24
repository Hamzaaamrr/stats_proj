# 03_risk_score.R - how risky is each driver? (Student C)
# Idea #1: a risk score per session and per driver, like fleet companies / insurers use.
source("01_load_data.R")

# ---- Settings (our own design choices - explain them in the presentation) ----
# How dangerous each behavior is (follows the app's priority order)
weights <- c("Normal" = 0, "Drowsy" = 3, "Using Phone" = 3,
             "Distracted" = 2, "Drinking" = 1, "Eating" = 1)

min_session_seconds <- 60   # shorter sessions are too short to judge fairly

# Risk score = weighted risky seconds / (3 x all seconds) x 100
#   0   = normal driving the whole time
#   100 = drowsy or on the phone the whole time
risk_band <- function(score) {
  ifelse(score < 10, "Low", ifelse(score <= 25, "Medium", "High"))
}


# ---- 1. Risk score per session -----------------------------------------------
# samples has one row per second, so each row = 1 second of driving.
long_sessions <- sessions$Session_ID[sessions$Duration_s >= min_session_seconds]

session_scores <- samples %>%
  filter(Session_ID %in% long_sessions) %>%
  mutate(Weight = weights[as.character(Behavior)]) %>%
  group_by(Session_ID, Driver_ID) %>%
  summarise(Minutes       = round(n() / 60, 1),
            Risky_Percent = round(100 * mean(Behavior != "Normal"), 1),
            Risk_Score    = round(100 * sum(Weight) / (3 * n()), 1),
            .groups = "drop")

# Risky events per hour (from events.csv)
events_per_session <- events %>%
  group_by(Session_ID) %>%
  summarise(Risky_Events = sum(Is_Risky))

session_scores <- session_scores %>%
  left_join(events_per_session, by = "Session_ID") %>%
  mutate(Risky_Events_per_Hour = round(Risky_Events / (Minutes / 60)),
         Band = risk_band(Risk_Score))

print(session_scores)
write_csv(session_scores, "output/tables/03_session_scores.csv")


# ---- 2. Risk score per driver (all their sessions together) -------------------
driver_scores <- samples %>%
  filter(Session_ID %in% long_sessions) %>%
  mutate(Weight = weights[as.character(Behavior)]) %>%
  group_by(Driver_ID) %>%
  summarise(Sessions   = n_distinct(Session_ID),
            Minutes    = round(n() / 60, 1),
            Risk_Score = round(100 * sum(Weight) / (3 * n()), 1)) %>%
  mutate(Band = risk_band(Risk_Score)) %>%
  arrange(desc(Risk_Score))                     # riskiest driver first

print(driver_scores)
write_csv(driver_scores, "output/tables/03_driver_scores.csv")

p1 <- ggplot(driver_scores, aes(x = reorder(Driver_ID, Risk_Score), y = Risk_Score, fill = Band)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = Risk_Score), hjust = -0.2) +
  coord_flip() +
  scale_fill_manual(values = c("Low" = "seagreen", "Medium" = "orange", "High" = "firebrick")) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Driver risk ranking",
       subtitle = "0 = always normal, 100 = drowsy or on the phone the whole time",
       x = "Driver", y = "Risk score (0-100)") +
  theme_minimal()
print(p1)
ggsave("output/figures/03_driver_ranking.png", p1, width = 8, height = 4, bg = "white")


# ---- 3. Risk score per session (compare trips) --------------------------------
p2 <- ggplot(session_scores, aes(x = Session_ID, y = Risk_Score, fill = Band)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = Risk_Score), vjust = -0.4) +
  facet_grid(~ Driver_ID, scales = "free_x", space = "free_x") +   # one panel per driver
  scale_fill_manual(values = c("Low" = "seagreen", "Medium" = "orange", "High" = "firebrick")) +
  labs(title = "Risk score of each session",
       subtitle = paste("Sessions shorter than", min_session_seconds, "s are left out"),
       x = "Session", y = "Risk score (0-100)") +
  theme_minimal()
print(p2)
ggsave("output/figures/03_session_scores.png", p2, width = 9, height = 5, bg = "white")


# ---- 4. What makes up each driver's score? ------------------------------------
# Points each behavior adds to the score (the bars add up to the driver's score).
breakdown <- samples %>%
  filter(Session_ID %in% long_sessions) %>%
  mutate(Weight = weights[as.character(Behavior)]) %>%
  group_by(Driver_ID) %>%
  mutate(Total_Seconds = n()) %>%
  group_by(Driver_ID, Behavior) %>%
  summarise(Points = round(100 * sum(Weight) / (3 * first(Total_Seconds)), 1), .groups = "drop") %>%
  filter(Behavior != "Normal")

print(breakdown)
write_csv(breakdown, "output/tables/03_score_breakdown.csv")

p3 <- ggplot(breakdown, aes(x = Driver_ID, y = Points, fill = Behavior)) +
  geom_col(width = 0.6) +
  scale_fill_manual(values = behavior_colors) +
  labs(title = "What makes up each driver's risk score",
       x = "Driver", y = "Points added to the risk score", fill = NULL) +
  theme_minimal()
print(p3)
ggsave("output/figures/03_score_breakdown.png", p3, width = 8, height = 5, bg = "white")

cat("\nDone. Charts are in output/figures, tables in output/tables.\n")
