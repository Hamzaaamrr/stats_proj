# 04_stat_tests.R - are the differences real or just chance? (Student D)
# Each test prints a plain-English answer. Rule: p < 0.05 means "a real difference".
source("01_load_data.R")

min_session_seconds <- 60   # same rule as the risk score (03)

# Compare drivers if we have 2+ drivers, otherwise compare sessions.
long_sessions <- sessions %>% filter(Duration_s >= min_session_seconds)
compare_by <- ifelse(n_distinct(long_sessions$Driver_ID) >= 2, "Driver_ID", "Session_ID")
cat("\nComparing by:", compare_by, "\n")

answer <- function(p, yes, no) {
  if (p < 0.05) paste0("YES (p = ", signif(p, 3), "): ", yes)
  else          paste0("NO  (p = ", signif(p, 3), "): ", no)
}


# ---- Test 1: Do groups have different rates of risky events? (Poisson) --------
# Counts of risky events per session, and how long each session was.
rates <- events %>%
  filter(Session_ID %in% long_sessions$Session_ID) %>%
  group_by(Session_ID, Driver_ID) %>%
  summarise(Risky_Events = sum(Is_Risky), .groups = "drop") %>%
  left_join(long_sessions %>% select(Session_ID, Duration_s), by = "Session_ID") %>%
  mutate(Minutes = Duration_s / 60, Group = .data[[compare_by]])

# Poisson model: events ~ group, with session length as the "exposure"
model <- glm(Risky_Events ~ Group + offset(log(Minutes)), family = poisson, data = rates)
p1 <- anova(model, test = "Chisq")$`Pr(>Chi)`[2]

result1 <- answer(p1, "the rate of risky events really differs between groups.",
                      "the differences in risky-event rates could be chance.")
cat("\nTest 1 - Different rates of risky events?\n ", result1, "\n")

# Chart: risky events per hour with a 95% confidence interval for each group
rate_table <- rates %>%
  group_by(Group) %>%
  summarise(Events = sum(Risky_Events), Hours = sum(Minutes) / 60) %>%
  rowwise() %>%
  mutate(Per_Hour = Events / Hours,
         Low  = poisson.test(Events, Hours)$conf.int[1],
         High = poisson.test(Events, Hours)$conf.int[2]) %>%
  ungroup()
print(rate_table)
write_csv(rate_table, "output/tables/04_risky_rates.csv")

g1 <- ggplot(rate_table, aes(x = Group, y = Per_Hour)) +
  geom_col(fill = "tomato", width = 0.6) +
  geom_errorbar(aes(ymin = Low, ymax = High), width = 0.2) +
  labs(title = "Risky events per hour (with 95% confidence interval)",
       subtitle = paste("Test 1:", result1),
       x = NULL, y = "Risky events per hour") +
  theme_minimal()
print(g1)
ggsave("output/figures/04_risky_rates.png", g1, width = 9, height = 5, bg = "white")


# ---- Test 2: Is the MIX of risky behaviors different? (Chi-square) ------------
mix <- events %>%
  filter(Is_Risky, Session_ID %in% long_sessions$Session_ID) %>%
  mutate(Group = .data[[compare_by]], Behavior = droplevels(Behavior))
mix_table <- table(mix$Group, mix$Behavior)
print(mix_table)

# Counts are small, so we use the simulated (Monte Carlo) version of the test
set.seed(1)
p2 <- chisq.test(mix_table, simulate.p.value = TRUE, B = 10000)$p.value
result2 <- answer(p2, "groups differ in WHICH risky behaviors they show.",
                      "the mix of risky behaviors is similar across groups.")
cat("\nTest 2 - Different mix of behaviors?\n ", result2, "\n")


# ---- Test 3: Do some behaviors last longer? (Kruskal-Wallis) ------------------
durations <- events %>%
  filter(Is_Risky) %>%
  group_by(Behavior) %>% filter(n() >= 3) %>% ungroup() %>%   # need a few events per behavior
  mutate(Behavior = droplevels(Behavior))

p3 <- kruskal.test(Duration_s ~ Behavior, data = durations)$p.value
result3 <- answer(p3, "some risky behaviors last longer than others.",
                      "risky behaviors last about the same time.")
cat("\nTest 3 - Do some behaviors last longer?\n ", result3, "\n")
print(durations %>% group_by(Behavior) %>%
        summarise(Events = n(), Median_s = median(Duration_s)))


# ---- Summary table -----------------------------------------------------------
summary_table <- data.frame(
  Test     = c("Poisson regression", "Chi-square", "Kruskal-Wallis"),
  Question = c(paste("Different risky-event rates between", compare_by, "?"),
               paste("Different mix of risky behaviors between", compare_by, "?"),
               "Do some risky behaviors last longer?"),
  p_value  = signif(c(p1, p2, p3), 3),
  Answer   = c(result1, result2, result3)
)
print(summary_table)
write_csv(summary_table, "output/tables/04_test_results.csv")

cat("\nDone. Note: with only", n_distinct(long_sessions$Driver_ID),
    "driver(s), these compare", compare_by, "- record more drivers to compare people.\n")
