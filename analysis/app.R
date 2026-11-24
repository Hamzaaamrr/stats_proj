# app.R - Driver Monitor dashboard (Shiny)
# Open analysis.Rproj, open this file, click "Run App" (top right of the editor).
# While the camera app runs, the Live tab updates every second.
library(shiny)
library(bslib)
library(dplyr)
library(readr)
library(ggplot2)

# ---- Shared settings (same as the analysis scripts) --------------------------
behaviors <- c("Normal", "Drowsy", "Using Phone", "Distracted", "Drinking", "Eating")
# Same colors as 01_load_data.R, written as hex codes so the web page understands them too
behavior_colors <- c("Normal" = "#999999", "Drowsy" = "#FF0000", "Using Phone" = "#FFA500",
                     "Distracted" = "#A020F0", "Drinking" = "#0000FF", "Eating" = "#006400")
weights <- c("Normal" = 0, "Drowsy" = 3, "Using Phone" = 3,
             "Distracted" = 2, "Drinking" = 1, "Eating" = 1)        # same as 03
band_colors <- c("Low" = "#2E8B57", "Medium" = "#FFA500", "High" = "#B22222")
risk_band <- function(score) ifelse(score < 10, "Low", ifelse(score <= 25, "Medium", "High"))
min_session_seconds <- 60

SAMPLES_FILE <- "../data/samples.csv"
EVENTS_FILE  <- "../data/events.csv"

# Read a CSV; timestamps are read as local time (needed for the "live" check).
read_data <- function(path) {
  if (!file.exists(path)) return(NULL)
  d <- read_csv(path, show_col_types = FALSE, locale = locale(tz = Sys.timezone()))
  d$Behavior  <- factor(d$Behavior, levels = behaviors)
  if ("True_Label" %in% names(d)) d$True_Label <- factor(d$True_Label, levels = behaviors)
  d
}

# Small "card" for the Live tab
state_card <- function(title, value, color = "#333333") {
  value_box(title = title, value = value,
            theme = value_box_theme(bg = color, fg = "white"))
}


# ---- Page layout -------------------------------------------------------------
ui <- page_navbar(
  title = "Driver Monitor",
  theme = bs_theme(bootswatch = "flatly"),

  nav_panel("Live",
    layout_columns(
      uiOutput("live_state"), uiOutput("live_alert"), uiOutput("live_ear"), uiOutput("live_head"),
      col_widths = c(3, 3, 3, 3)
    ),
    textOutput("live_info"),
    card(card_header("Timeline of the current session"), plotOutput("live_timeline", height = 210)),
    card(card_header("Most recent events"), tableOutput("live_events"))
  ),

  nav_panel("Session Report",
    layout_sidebar(
      sidebar = sidebar(selectInput("session", "Session", choices = NULL),
                        tableOutput("session_info")),
      card(card_header("Timeline"), plotOutput("session_timeline", height = 210)),
      layout_columns(
        card(card_header("Share of time in each behavior"), plotOutput("session_share")),
        card(card_header("How long risky events lasted (seconds)"), tableOutput("session_durations"))
      ),
      card(card_header("Confidence per behavior"), plotOutput("session_confidence", height = 280))
    )
  ),

  nav_panel("Driver Risk",
    p("Risk score 0-100: 0 = always normal, 100 = drowsy or on the phone the whole time.",
      "Weights: Drowsy 3, Phone 3, Distracted 2, Drinking 1, Eating 1.",
      paste0("Sessions shorter than ", min_session_seconds, " s are left out.")),
    layout_columns(
      card(card_header("Driver ranking"), plotOutput("driver_ranking", height = 300)),
      card(card_header("What makes up each driver's score"), plotOutput("driver_breakdown", height = 300))
    ),
    card(card_header("Risk score of each session"), tableOutput("session_scores"))
  ),

  nav_panel("Accuracy",
    p("Test-mode sessions only: what the app said vs what the driver was really doing."),
    layout_columns(uiOutput("acc_strict"), uiOutput("acc_tolerant"), uiOutput("acc_seconds"),
                   col_widths = c(4, 4, 4)),
    layout_columns(
      card(card_header("Confusion matrix (row %)"), plotOutput("confusion", height = 380)),
      card(card_header("Precision and recall per behavior"), tableOutput("precision_recall"))
    )
  )
)


# ---- What the page shows -----------------------------------------------------
server <- function(input, output, session) {

  # Re-read the files every second (only when they changed on disk)
  samples <- reactiveFileReader(1000, session, SAMPLES_FILE, read_data)
  events  <- reactiveFileReader(1000, session, EVENTS_FILE,  read_data)

  # ===== Live tab =====
  latest_session <- reactive({
    s <- samples(); req(s, nrow(s) > 0)
    s %>% filter(Session_ID == last(Session_ID))
  })
  latest_row <- reactive(tail(latest_session(), 1))
  is_live <- reactive({
    as.numeric(difftime(Sys.time(), latest_row()$Timestamp, units = "secs")) < 5
  })

  output$live_state <- renderUI({
    r <- latest_row(); b <- as.character(r$Behavior)
    state_card("Driver state", b, behavior_colors[[b]])
  })
  output$live_alert <- renderUI({
    r <- latest_row()
    state_card("Warning", r$Alert, if (r$Alert == "ON") "#B22222" else "#2E8B57")
  })
  output$live_ear <- renderUI({
    r <- latest_row()
    state_card("Eye openness (EAR)", ifelse(is.na(r$EAR), "no face", sprintf("%.3f", r$EAR)), "#2c3e50")
  })
  output$live_head <- renderUI(state_card("Head position", latest_row()$Head_Pose, "#2c3e50"))

  output$live_info <- renderText({
    invalidateLater(1000)                                   # keep the "seconds ago" fresh
    r <- latest_row()
    status <- if (is_live()) "LIVE" else "No session running - showing the last one"
    paste0(status, "  |  Session ", r$Session_ID, ", driver ", r$Driver_ID,
           "  |  ", round(r$Time_s / 60, 1), " min into the trip",
           "  |  last update ", format(r$Timestamp, "%H:%M:%S"))
  })

  output$live_timeline <- renderPlot(timeline_plot(latest_session()))

  output$live_events <- renderTable({
    e <- events(); req(e)
    e %>% filter(Session_ID == latest_row()$Session_ID) %>%
      tail(8) %>% arrange(desc(Event_ID)) %>%
      transmute(Time = format(Timestamp, "%H:%M:%S"), Behavior = as.character(Behavior),
                Seconds = Duration_s, Trigger, Confidence)
  }, na = "")

  # ===== Session Report tab =====
  observe({
    ids <- unique(samples()$Session_ID)
    updateSelectInput(session, "session", choices = rev(ids),
                      selected = if (input$session %in% ids) input$session else last(ids))
  })
  chosen <- reactive({ req(input$session); samples() %>% filter(Session_ID == input$session) })

  output$session_info <- renderTable({
    s <- chosen()
    data.frame(Item = c("Driver", "Length (min)", "Risky time"),
               Value = c(s$Driver_ID[1], round(nrow(s) / 60, 1),
                         paste0(round(100 * mean(s$Behavior != "Normal")), "%")))
  })
  output$session_timeline <- renderPlot(timeline_plot(chosen()))
  output$session_share <- renderPlot({
    chosen() %>% count(Behavior) %>% mutate(Percent = 100 * n / sum(n)) %>%
      ggplot(aes(x = Behavior, y = Percent, fill = Behavior)) +
      geom_col(show.legend = FALSE) +
      geom_text(aes(label = paste0(round(Percent), "%")), vjust = -0.3) +
      scale_fill_manual(values = behavior_colors) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +   # room for the labels
      labs(x = NULL, y = "% of session time") + theme_minimal(base_size = 13)
  })
  output$session_durations <- renderTable({
    e <- events(); req(e)
    e %>% filter(Session_ID == input$session, Behavior != "Normal") %>%
      group_by(Behavior) %>%
      summarise(Events = n(), Average = mean(Duration_s), Longest = max(Duration_s))
  }, digits = 1)
  output$session_confidence <- renderPlot({
    e <- events(); req(e)
    e %>% filter(Session_ID == input$session) %>%
      ggplot(aes(x = Behavior, y = Confidence, fill = Behavior)) +
      geom_boxplot(show.legend = FALSE) +
      scale_fill_manual(values = behavior_colors) +
      labs(x = NULL, y = "Confidence (0-1)") + theme_minimal(base_size = 13)
  })

  # ===== Driver Risk tab (same formula as 03_risk_score.R) =====
  scored <- reactive({
    samples() %>%
      group_by(Session_ID) %>% filter(n() >= min_session_seconds) %>% ungroup() %>%
      mutate(Weight = weights[as.character(Behavior)])
  })
  output$driver_ranking <- renderPlot({
    d <- scored() %>% group_by(Driver_ID) %>%
      summarise(Risk_Score = round(100 * sum(Weight) / (3 * n()), 1)) %>%
      mutate(Band = risk_band(Risk_Score))
    ggplot(d, aes(x = reorder(Driver_ID, Risk_Score), y = Risk_Score, fill = Band)) +
      geom_col(width = 0.6) + geom_text(aes(label = Risk_Score), hjust = -0.2) +
      coord_flip() + scale_fill_manual(values = band_colors) +
      scale_y_continuous(limits = c(0, 100)) +
      labs(x = "Driver", y = "Risk score (0-100)") + theme_minimal(base_size = 13)
  })
  output$driver_breakdown <- renderPlot({
    scored() %>% group_by(Driver_ID) %>% mutate(Total = n()) %>%
      group_by(Driver_ID, Behavior) %>%
      summarise(Points = 100 * sum(Weight) / (3 * first(Total)), .groups = "drop") %>%
      filter(Behavior != "Normal") %>%
      ggplot(aes(x = Driver_ID, y = Points, fill = Behavior)) +
      geom_col(width = 0.6) + scale_fill_manual(values = behavior_colors) +
      labs(x = "Driver", y = "Points added to the score", fill = NULL) + theme_minimal(base_size = 13)
  })
  output$session_scores <- renderTable({
    scored() %>% group_by(Session_ID, Driver_ID) %>%
      summarise(Minutes = n() / 60,
                Risky_Percent = 100 * mean(Behavior != "Normal"),
                Risk_Score = 100 * sum(Weight) / (3 * n()), .groups = "drop") %>%
      mutate(Band = risk_band(Risk_Score)) %>% arrange(desc(Risk_Score))
  }, digits = 1)

  # ===== Accuracy tab (same method as 05_accuracy.R) =====
  test <- reactive({
    same <- function(a, b) coalesce(a == b, FALSE)
    samples() %>% filter(!is.na(True_Label)) %>% arrange(Session_ID, Time_s) %>%
      group_by(Session_ID) %>%
      mutate(Correct = same(Behavior, True_Label),
             Correct_Tolerant = Correct |
               same(Behavior, lag(True_Label, 1))  | same(Behavior, lag(True_Label, 2)) |
               same(Behavior, lead(True_Label, 1)) | same(Behavior, lead(True_Label, 2))) %>%
      ungroup()
  })
  output$acc_strict   <- renderUI(state_card("Accuracy (strict)", paste0(round(100 * mean(test()$Correct), 1), "%"), "#2c3e50"))
  output$acc_tolerant <- renderUI(state_card("Accuracy (+/- 2 s)", paste0(round(100 * mean(test()$Correct_Tolerant), 1), "%"), "#4682B4"))
  output$acc_seconds  <- renderUI(state_card("Labeled seconds", nrow(test()), "#777777"))

  output$confusion <- renderPlot({
    as.data.frame(table(True = test()$True_Label, Predicted = test()$Behavior)) %>%
      group_by(True) %>% mutate(Row_Total = sum(Freq)) %>% ungroup() %>%
      filter(Row_Total > 0) %>% mutate(Row_Percent = round(100 * Freq / Row_Total)) %>%
      ggplot(aes(x = Predicted, y = True, fill = Row_Percent)) +
      geom_tile(color = "white") +
      geom_text(aes(label = paste0(Row_Percent, "%\n(", Freq, ")")), size = 3.5) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      scale_y_discrete(limits = rev) +
      labs(x = "App said", y = "Really happened", fill = "% of row") +
      theme_minimal(base_size = 13) + theme(axis.text.x = element_text(angle = 30, hjust = 1))
  })
  output$precision_recall <- renderTable({
    t <- test()
    data.frame(Behavior = behaviors) %>% rowwise() %>%
      mutate(Real_Seconds = sum(t$True_Label == Behavior),
             Precision = 100 * sum(t$True_Label == Behavior & t$Behavior == Behavior) / sum(t$Behavior == Behavior),
             Recall    = 100 * sum(t$True_Label == Behavior & t$Behavior == Behavior) / Real_Seconds) %>%
      ungroup()
  }, digits = 1, na = "n/a")
}

# Timeline strip used by the Live and Session Report tabs
timeline_plot <- function(s) {
  ggplot(s, aes(x = Time_s / 60, y = 1, fill = Behavior)) +
    geom_tile(width = 1.1 / 60) +
    scale_fill_manual(values = behavior_colors) +
    labs(x = "Minutes into the session", y = NULL, fill = NULL) +
    theme_minimal(base_size = 13) +
    theme(axis.text.y = element_blank(), panel.grid = element_blank(), legend.position = "right")
}

shinyApp(ui, server)
