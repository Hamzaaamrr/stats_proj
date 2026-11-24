# 01_load_data.R - load the app's data (Student A)
library(dplyr)
library(readr)
library(ggplot2)

# 1. Read the three files the app writes
sessions <- read_csv("../data/sessions.csv")
events   <- read_csv("../data/events.csv")
samples  <- read_csv("../data/samples.csv")

# 2. Put the behaviors in a fixed order (so every chart lists them the same way)
behaviors <- c("Normal", "Drowsy", "Using Phone", "Distracted", "Drinking", "Eating")
events$Behavior     <- factor(events$Behavior,     levels = behaviors)
samples$Behavior    <- factor(samples$Behavior,    levels = behaviors)
samples$True_Label  <- factor(samples$True_Label,  levels = behaviors)

# 3. Helpful extra columns
events$Is_Risky   <- events$Behavior != "Normal"
samples$Minute    <- samples$Time_s / 60
samples$Correct   <- samples$Behavior == samples$True_Label   # only in test mode
samples$Version   <- ifelse(is.na(samples$Object_At_Face), "before fix", "after fix")

# 4. One color per behavior, used in every chart
behavior_colors <- c("Normal" = "grey60", "Drowsy" = "red", "Using Phone" = "orange",
                     "Distracted" = "purple", "Drinking" = "blue", "Eating" = "darkgreen")

# 5. Quick check
print(table(events$Behavior))
print(sessions[, c("Session_ID", "Driver_ID", "Duration_s")])