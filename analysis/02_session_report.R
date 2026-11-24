# 02_session_report.R - what happened in each trip (Student B)
# Idea #6: a "trip report" per session.
# Charts are saved in output/figures, tables in output/tables (names start with 02_).
source("01_load_data.R")

# Change this to pick which session the timeline (chart 4) shows
timeline_session <- "S05"


# ---- 1. Share of time in each behavior, per session --------------------------
# samples.csv has one row per second, so counting rows = counting seconds.
time_share <- samples %>%
  count(Session_ID, Behavior) %>%
  group_by(Session_ID) %>%
  mutate(Seconds = n, Percent = round(100 * n / sum(n), 1)) %>%
  ungroup() %>%
  select(Session_ID, Behavior, Seconds, Percent)

print(time_share)
write_csv(time_share, "output/tables/02_time_share.csv")

p1 <- ggplot(time_share, aes(x = Session_ID, y = Percent, fill = Behavior)) +
  geom_col() +
  scale_fill_manual(values = behavior_colors) +
  labs(title = "Share of each trip spent in each behavior",
       x = "Session", y = "% of session time", fill = NULL) +
  theme_minimal()
print(p1)
ggsave("output/figures/02_time_share.png", p1, width = 8, height = 5, bg = "white")


# ---- 2. Number of events per behavior ----------------------------------------
event_counts <- events %>%
  count(Behavior, .drop = FALSE) %>%           # .drop = FALSE keeps behaviors with 0 events
  mutate(Percent = round(100 * n / sum(n), 1))

print(event_counts)
write_csv(event_counts, "output/tables/02_event_counts.csv")

p2 <- ggplot(event_counts, aes(x = Behavior, y = n, fill = Behavior)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = n), vjust = -0.4) +
  scale_fill_manual(values = behavior_colors) +
  labs(title = "Number of detected events per behavior (all sessions)",
       x = NULL, y = "Events") +
  theme_minimal()
print(p2)
ggsave("output/figures/02_event_counts.png", p2, width = 8, height = 5, bg = "white")


# ---- 3. How long do risky events last? ---------------------------------------
risky_events <- events %>% filter(Is_Risky)

durations <- risky_events %>%
  group_by(Behavior) %>%
  summarise(Events     = n(),
            Mean_s     = round(mean(Duration_s), 1),
            Median_s   = round(median(Duration_s), 1),
            Longest_s  = round(max(Duration_s), 1),
            Total_s    = round(sum(Duration_s), 1))

print(durations)
write_csv(durations, "output/tables/02_durations.csv")

p3 <- ggplot(risky_events, aes(x = Behavior, y = Duration_s, fill = Behavior)) +
  geom_boxplot(show.legend = FALSE, outlier.shape = NA, alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.6, show.legend = FALSE) +   # each dot = one event
  scale_fill_manual(values = behavior_colors) +
  labs(title = "How long risky events last",
       subtitle = "Each dot is one event; the box shows the middle 50%",
       x = NULL, y = "Duration (seconds)") +
  theme_minimal()
print(p3)
ggsave("output/figures/02_durations.png", p3, width = 8, height = 5, bg = "white")


# ---- 4. Timeline of one trip --------------------------------------------------
one_trip <- samples %>% filter(Session_ID == timeline_session)

p4 <- ggplot(one_trip, aes(x = Minute, y = 1, fill = Behavior)) +
  geom_tile(width = 1.1 / 60, height = 1) +     # each row is ~1 second = 1/60 of a minute (+10% so no gaps)
  scale_fill_manual(values = behavior_colors) +
  labs(title = paste("Timeline of session", timeline_session),
       x = "Minutes into the session", y = NULL, fill = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_blank(), panel.grid = element_blank(),
        legend.position = "bottom")
print(p4)
ggsave(paste0("output/figures/02_timeline_", timeline_session, ".png"), p4,
       width = 9, height = 2.5, bg = "white")


# ---- 5. How confident was the app? -------------------------------------------
p5 <- ggplot(events, aes(x = Behavior, y = Confidence, fill = Behavior)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = behavior_colors) +
  labs(title = "Confidence scores per behavior",
       x = NULL, y = "Confidence (0-1)") +
  theme_minimal()
print(p5)
ggsave("output/figures/02_confidence.png", p5, width = 8, height = 5, bg = "white")


# ---- 6. When in a trip do risky behaviors happen? ----------------------------
# Group the seconds into whole minutes (0 = first minute) and count how many were risky.
by_minute <- samples %>%
  mutate(Minute_Bin = floor(Minute)) %>%
  group_by(Minute_Bin) %>%
  summarise(Seconds = n(),
            Risky_Percent = round(100 * mean(Behavior != "Normal"), 1))

print(by_minute)
write_csv(by_minute, "output/tables/02_risky_by_minute.csv")

p6 <- ggplot(by_minute, aes(x = Minute_Bin, y = Risky_Percent)) +
  geom_col(fill = "tomato") +
  geom_text(aes(label = paste0("n=", Seconds)), vjust = -0.4, size = 3) +
  labs(title = "Share of risky time by minute into the trip (all sessions)",
       subtitle = "n = seconds of data in that minute (later minutes have fewer sessions)",
       x = "Minute of the session (0 = first minute)", y = "% of seconds that were risky") +
  theme_minimal()
print(p6)
ggsave("output/figures/02_risky_by_minute.png", p6, width = 8, height = 5, bg = "white")

cat("\nDone. Charts are in output/figures, tables in output/tables.\n")
