#############################################################################
# Wine Quality Explorer — an R Shiny app for the "Data Science Life Cycle"
# case study.
#
# Dataset : UCI Machine Learning Repository — Wine Quality (Red Wine)
#           P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis, 2009.
#           https://archive.ics.uci.edu/ml/datasets/wine+quality
#
# Task    : Predict wine quality (sensory score, 0-10) from 11 physicochemical
#           lab measurements.
#             - Regression:      Random Forest Regressor  vs. Linear Regression
#             - Classification:  Random Forest Classifier ("good" wine, quality
#                                >= a user chosen cutoff, vs. "not good")
#
# Author  : Michael Creighton
#############################################################################

library(shiny)
library(shinythemes)
library(randomForest)
library(caret)
library(ggplot2)
library(reshape2)
library(DT)

set.seed(42)

## ---------------------------------------------------------------------
## 1. DATA
## ---------------------------------------------------------------------
wine <- read.csv("data/winequality-red.csv", stringsAsFactors = FALSE)

feature_names <- setdiff(names(wine), "quality")

feature_ranges <- lapply(wine[feature_names], function(x) {
  list(min = round(min(x), 3), max = round(max(x), 3), mean = round(mean(x), 3))
})

## R turns "fixed acidity" into "fixed.acidity" on read.csv(); keep a lookup
## of pretty labels for display purposes only (the underlying column names,
## used in formulas/inputs, are left untouched).
pretty_label <- function(x) gsub("\\.", " ", x)
feature_choices <- setNames(feature_names, sapply(feature_names, pretty_label))

## ---------------------------------------------------------------------
## 2. UI
## ---------------------------------------------------------------------
ui <- navbarPage(
  title = "Wine Quality Explorer",
  theme = shinytheme("flatly"),

  ## ---------------- TAB 1: ABOUT / DATA CYCLE -------------------------
  tabPanel("About & Methodology",
    fluidPage(
      h2("Wine Quality Explorer"),
      p("This application walks through the full data-science life cycle —",
        strong("problem definition -> data collection -> data preparation -> "),
        strong("modeling -> evaluation -> deployment"),
        " — using the UCI ", em("Wine Quality (Red Wine)"), " dataset."),

      h3("1. Problem / Motivation"),
      p("Wine producers currently rely on human sensory panels to grade wine
         quality, which is slow, expensive, and subjective. The goal here is
         to see whether objective, cheaply-measured physicochemical lab
         properties (acidity, sugar, sulfates, alcohol, etc.) can predict the
         quality score a panel would assign. This is interesting because it
         shows how a data-driven model could give producers fast, repeatable
         quality feedback during production."),

      h3("2. Data"),
      p("Source: UCI Machine Learning Repository, ", em("Wine Quality"),
        " data set (Cortez et al., 2009). 1,599 red wine samples, 11
        physicochemical input features, and 1 output — a median sensory
        quality score (0-10) given by at least 3 wine experts."),
      tags$ul(
        tags$li("fixed acidity, volatile acidity, citric acid"),
        tags$li("residual sugar, chlorides"),
        tags$li("free sulfur dioxide, total sulfur dioxide"),
        tags$li("density, pH, sulphates, alcohol"),
        tags$li(strong("quality"), " (target, integer 3-8 in this sample)")
      ),

      h3("3. Algorithm — Random Forest"),
      p("Random Forest is an ensemble of decision trees built with two
         sources of randomness: (a) each tree is trained on a ", em("bootstrap"),
        " resample of the training rows (bagging), and (b) at every split
         only a random subset of ", code("mtry"), " candidate features is
         considered rather than all of them. Averaging (regression) or
         majority-voting (classification) across many de-correlated trees
         reduces variance compared to a single tree, without increasing bias
         much."),
      withMathJax(
        p("For regression, each tree partitions the feature space into
           regions \\(R_1,\\dots,R_J\\) by greedily choosing, at each node,
           the split \\((\\,j^*, s^*)\\) that minimizes the sum of squared
           errors:"),
        p("$$ \\min_{j,\\,s} \\left[ \\sum_{x_i \\in R_1(j,s)} (y_i - \\bar y_{R_1})^2
             \\;+\\; \\sum_{x_i \\in R_2(j,s)} (y_i - \\bar y_{R_2})^2 \\right] $$"),
        p("For classification, the split criterion is the Gini impurity of
           node \\(m\\) with classes \\(k=1,\\dots,K\\):"),
        p("$$ G_m = \\sum_{k=1}^{K} \\hat p_{mk}(1-\\hat p_{mk}) $$"),
        p("The forest prediction is the average (regression) or majority
           vote (classification) of \\(B\\) trees:"),
        p("$$ \\hat f_{RF}(x) = \\frac{1}{B}\\sum_{b=1}^{B} T_b(x)
             \\qquad \\text{or} \\qquad
             \\hat y(x) = \\text{mode}\\{T_b(x)\\}_{b=1}^{B} $$"),
        p("Because roughly one third of rows are left out of each
           bootstrap sample (the \\(\\textit{out-of-bag}\\), or OOB, rows),
           Random Forest gets an almost-free, unbiased estimate of test
           error without needing a separate validation set — the app reports
           this OOB error alongside a held-out test split."),
        p("Variable importance is computed as the total decrease in node
           impurity (Gini for classification, RSS for regression)
           contributed by a feature, summed over all trees and splits, or
           via permutation: the increase in error when a feature's values
           are randomly shuffled in the OOB data.")
      ),
      p("The app also fits an ordinary Least-Squares Linear Regression as a
         simple, interpretable baseline to compare against Random Forest."),

      h3("4. How to use this app"),
      tags$ol(
        tags$li("Go to ", strong("Explore Data"), " to see distributions and
                 correlations between the physicochemical properties."),
        tags$li("Go to ", strong("Train Model"), ", choose Regression or
                 Classification, tune the Random Forest hyper-parameters
                 (number of trees, mtry, train/test split), and press
                 'Train Model' to see performance metrics and importance
                 plots update live."),
        tags$li("Go to ", strong("Predict"), " and move the sliders to enter
                 your own physicochemical measurements to get a live
                 predicted quality score / class from the currently trained
                 model.")
      ),
      hr(),
      p(em("Data-science life cycle checklist: business question -> data
            collection (UCI) -> cleaning/EDA (Explore tab) -> feature/target
            split & modeling (Train tab) -> evaluation (metrics + OOB) ->
            deployment (this Shiny app on shinyapps.io) -> interpretation
            (variable importance, prediction tab)."))
    )
  ),

  ## ---------------- TAB 2: EXPLORE DATA --------------------------------
  tabPanel("Explore Data",
    sidebarLayout(
      sidebarPanel(
        h4("Data Exploration Controls"),
        selectInput("hist_var", "Variable to inspect:",
                    choices = feature_choices, selected = "alcohol"),
        sliderInput("bins", "Number of histogram bins:",
                    min = 5, max = 60, value = 30),
        checkboxInput("show_by_quality", "Color by quality score", TRUE),
        hr(),
        p("Use this tab to understand the raw data before modeling —
           step 3 of the data-science life cycle (EDA).")
      ),
      mainPanel(
        h4("Summary statistics"),
        verbatimTextOutput("summary_stats"),
        plotOutput("hist_plot", height = "320px"),
        h4("Correlation matrix"),
        plotOutput("corr_plot", height = "420px"),
        h4("Raw data (first 200 rows)"),
        DTOutput("data_table")
      )
    )
  ),

  ## ---------------- TAB 3: TRAIN MODEL ----------------------------------
  tabPanel("Train Model",
    sidebarLayout(
      sidebarPanel(
        h4("Modeling Task"),
        radioButtons("task", "Task type:",
                     choices = c("Regression (predict quality 0-10)" = "reg",
                                 "Classification (good vs. not-good wine)" = "clf")),
        conditionalPanel(
          condition = "input.task == 'clf'",
          sliderInput("good_cutoff", "‘Good wine’ cutoff (quality >= this value):",
                      min = 5, max = 8, value = 7, step = 1)
        ),
        hr(),
        h4("Random Forest Hyper-parameters"),
        sliderInput("ntree", "Number of trees (ntree):",
                    min = 50, max = 1000, value = 300, step = 50),
        sliderInput("mtry", "Features considered per split (mtry):",
                    min = 1, max = length(feature_names),
                    value = 4, step = 1),
        sliderInput("split", "Training set proportion:",
                    min = 0.5, max = 0.9, value = 0.75, step = 0.05),
        checkboxInput("compare_lm", "Compare with Linear Regression baseline
                       (regression task only)", TRUE),
        actionButton("train_btn", "Train Model", class = "btn-primary"),
        hr(),
        p("Random Forest is retrained on demand so you can see how the
           hyper-parameters change test-set performance and OOB error.")
      ),
      mainPanel(
        h4("Model Performance"),
        verbatimTextOutput("model_metrics"),
        conditionalPanel(
          condition = "input.task == 'reg'",
          h4("Predicted vs. Actual (test set)"),
          plotOutput("pred_vs_actual", height = "350px")
        ),
        conditionalPanel(
          condition = "input.task == 'clf'",
          h4("Confusion Matrix (test set)"),
          plotOutput("conf_matrix_plot", height = "350px")
        ),
        h4("Variable Importance"),
        plotOutput("var_importance", height = "350px"),
        h4("OOB Error vs. Number of Trees"),
        plotOutput("oob_plot", height = "300px")
      )
    )
  ),

  ## ---------------- TAB 4: PREDICT --------------------------------------
  tabPanel("Predict",
    sidebarLayout(
      sidebarPanel(
        h4("Enter physicochemical measurements"),
        uiOutput("predict_sliders"),
        hr(),
        p("Sliders default to the dataset mean. Predictions use the model
           currently trained in the 'Train Model' tab.")
      ),
      mainPanel(
        h3("Prediction"),
        htmlOutput("prediction_output"),
        hr(),
        p("Note: train the model in the 'Train Model' tab first (a default
           model is trained automatically on app start).")
      )
    )
  )
)

## ---------------------------------------------------------------------
## 3. SERVER
## ---------------------------------------------------------------------
server <- function(input, output, session) {

  ## ---------- Explore Data tab ----------
  output$summary_stats <- renderPrint({
    summary(wine[[input$hist_var]])
  })

  output$hist_plot <- renderPlot({
    df <- wine
    df$quality_f <- factor(df$quality)
    p <- ggplot(df, aes_string(x = input$hist_var)) 
    if (input$show_by_quality) {
      p <- p + geom_histogram(aes(fill = quality_f), bins = input$bins,
                               color = "white", alpha = 0.9) +
        labs(fill = "Quality")
    } else {
      p <- p + geom_histogram(bins = input$bins, fill = "#7b1e3a",
                               color = "white", alpha = 0.9)
    }
    p + theme_minimal(base_size = 14) +
      labs(title = paste("Distribution of", pretty_label(input$hist_var)),
           x = pretty_label(input$hist_var), y = "Count")
  })

  output$corr_plot <- renderPlot({
    cm <- round(cor(wine), 2)
    melted <- melt(cm)
    ggplot(melted, aes(Var1, Var2, fill = value)) +
      geom_tile() +
      geom_text(aes(label = value), size = 3) +
      scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                            midpoint = 0, limits = c(-1, 1)) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "", y = "", fill = "Corr")
  })

  output$data_table <- renderDT({
    datatable(head(wine, 200), options = list(pageLength = 10, scrollX = TRUE))
  })

  ## ---------- Train Model tab ----------
  model_result <- eventReactive(input$train_btn, {
    withProgress(message = "Training model...", value = 0.3, {

      set.seed(123)
      n <- nrow(wine)
      train_idx <- sample(seq_len(n), size = floor(input$split * n))

      if (input$task == "reg") {
        train_df <- wine[train_idx, ]
        test_df  <- wine[-train_idx, ]

        rf <- randomForest(quality ~ ., data = train_df,
                            ntree = input$ntree, mtry = input$mtry,
                            importance = TRUE)
        pred_rf <- predict(rf, test_df)
        rmse_rf <- sqrt(mean((pred_rf - test_df$quality)^2))
        r2_rf   <- 1 - sum((pred_rf - test_df$quality)^2) /
                       sum((test_df$quality - mean(test_df$quality))^2)

        lm_fit <- lm_r2 <- lm_rmse <- pred_lm <- NULL
        if (input$compare_lm) {
          lm_fit <- lm(quality ~ ., data = train_df)
          pred_lm <- predict(lm_fit, test_df)
          lm_rmse <- sqrt(mean((pred_lm - test_df$quality)^2))
          lm_r2   <- 1 - sum((pred_lm - test_df$quality)^2) /
                          sum((test_df$quality - mean(test_df$quality))^2)
        }

        incProgress(0.7)
        list(task = "reg", rf = rf, test_df = test_df, pred_rf = pred_rf,
             rmse_rf = rmse_rf, r2_rf = r2_rf,
             lm_fit = lm_fit, pred_lm = pred_lm, lm_rmse = lm_rmse, lm_r2 = lm_r2)

      } else {
        wine2 <- wine
        wine2$good <- factor(ifelse(wine2$quality >= input$good_cutoff, "good", "not_good"))
        wine2$quality <- NULL
        train_df <- wine2[train_idx, ]
        test_df  <- wine2[-train_idx, ]

        rf <- randomForest(good ~ ., data = train_df,
                            ntree = input$ntree, mtry = input$mtry,
                            importance = TRUE)
        pred_rf <- predict(rf, test_df)
        acc <- mean(pred_rf == test_df$good)
        cm <- table(Predicted = pred_rf, Actual = test_df$good)

        incProgress(0.7)
        list(task = "clf", rf = rf, test_df = test_df, pred_rf = pred_rf,
             acc = acc, cm = cm)
      }
    })
  }, ignoreNULL = FALSE)  # trains once automatically on app start

  output$model_metrics <- renderPrint({
    res <- model_result()
    if (res$task == "reg") {
      cat("Random Forest Regression\n")
      cat("  Test RMSE :", round(res$rmse_rf, 4), "\n")
      cat("  Test R^2  :", round(res$r2_rf, 4), "\n")
      cat("  OOB MSE   :", round(tail(res$rf$mse, 1), 4), "\n\n")
      if (!is.null(res$lm_fit)) {
        cat("Linear Regression baseline\n")
        cat("  Test RMSE :", round(res$lm_rmse, 4), "\n")
        cat("  Test R^2  :", round(res$lm_r2, 4), "\n")
      }
    } else {
      cat("Random Forest Classification\n")
      cat("  Test Accuracy :", round(res$acc, 4), "\n")
      cat("  OOB error rate:", round(tail(res$rf$err.rate[, "OOB"], 1), 4), "\n\n")
      cat("Confusion matrix (rows = predicted, cols = actual):\n")
      print(res$cm)
    }
  })

  output$pred_vs_actual <- renderPlot({
    res <- model_result()
    req(res$task == "reg")
    df <- data.frame(actual = res$test_df$quality, predicted_rf = res$pred_rf)
    p <- ggplot(df, aes(x = actual, y = predicted_rf)) +
      geom_jitter(width = 0.15, height = 0.15, alpha = 0.5, color = "#7b1e3a") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray30") +
      theme_minimal(base_size = 14) +
      labs(x = "Actual quality", y = "Predicted quality (Random Forest)")
    if (!is.null(res$pred_lm)) {
      df$predicted_lm <- res$pred_lm
      p <- ggplot(df, aes(x = actual)) +
        geom_jitter(aes(y = predicted_rf, color = "Random Forest"), width = 0.15, height = 0.15, alpha = 0.5) +
        geom_jitter(aes(y = predicted_lm, color = "Linear Regression"), width = 0.15, height = 0.15, alpha = 0.5) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray30") +
        scale_color_manual(values = c("Random Forest" = "#7b1e3a", "Linear Regression" = "#2166ac")) +
        theme_minimal(base_size = 14) +
        labs(x = "Actual quality", y = "Predicted quality", color = "Model")
    }
    p
  })

  output$conf_matrix_plot <- renderPlot({
    res <- model_result()
    req(res$task == "clf")
    cm_df <- as.data.frame(res$cm)
    ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), size = 6, color = "white") +
      scale_fill_gradient(low = "#deebf7", high = "#08519c") +
      theme_minimal(base_size = 14) +
      labs(title = "Confusion Matrix")
  })

  output$var_importance <- renderPlot({
    res <- model_result()
    imp <- importance(res$rf)
    metric_col <- if ("%IncMSE" %in% colnames(imp)) "%IncMSE" else "MeanDecreaseAccuracy"
    imp_df <- data.frame(feature = pretty_label(rownames(imp)), importance = imp[, metric_col])
    imp_df <- imp_df[order(imp_df$importance), ]
    imp_df$feature <- factor(imp_df$feature, levels = imp_df$feature)
    ggplot(imp_df, aes(x = feature, y = importance)) +
      geom_col(fill = "#7b1e3a") +
      coord_flip() +
      theme_minimal(base_size = 14) +
      labs(x = "", y = "Importance (permutation)", title = "Random Forest Variable Importance")
  })

  output$oob_plot <- renderPlot({
    res <- model_result()
    if (res$task == "reg") {
      df <- data.frame(trees = seq_along(res$rf$mse), oob_mse = res$rf$mse)
      ggplot(df, aes(x = trees, y = oob_mse)) +
        geom_line(color = "#7b1e3a", linewidth = 1) +
        theme_minimal(base_size = 14) +
        labs(x = "Number of trees", y = "OOB MSE")
    } else {
      df <- data.frame(trees = seq_len(nrow(res$rf$err.rate)), oob_err = res$rf$err.rate[, "OOB"])
      ggplot(df, aes(x = trees, y = oob_err)) +
        geom_line(color = "#7b1e3a", linewidth = 1) +
        theme_minimal(base_size = 14) +
        labs(x = "Number of trees", y = "OOB error rate")
    }
  })

  ## ---------- Predict tab ----------
  output$predict_sliders <- renderUI({
    lapply(feature_names, function(f) {
      rng <- feature_ranges[[f]]
      sliderInput(paste0("pred_", f), pretty_label(f),
                  min = rng$min, max = rng$max, value = rng$mean,
                  step = signif((rng$max - rng$min) / 100, 2))
    })
  })

  output$prediction_output <- renderUI({
    res <- model_result()
    new_row <- as.data.frame(lapply(feature_names, function(f) {
      val <- input[[paste0("pred_", f)]]
      if (is.null(val)) feature_ranges[[f]]$mean else val
    }))
    names(new_row) <- feature_names

    if (res$task == "reg") {
      pred <- predict(res$rf, new_row)
      extra <- ""
      if (!is.null(res$lm_fit)) {
        pred_lm <- predict(res$lm_fit, new_row)
        extra <- paste0("<p>Linear Regression prediction: <b>", round(pred_lm, 2), "</b></p>")
      }
      HTML(paste0("<h2 style='color:#7b1e3a'>Predicted quality (Random Forest): ",
                   round(pred, 2), " / 10</h2>", extra))
    } else {
      pred <- predict(res$rf, new_row)
      prob <- predict(res$rf, new_row, type = "prob")
      HTML(paste0("<h2 style='color:#7b1e3a'>Predicted class: ", as.character(pred), "</h2>",
                   "<p>P(good) = ", round(prob[1, "good"], 3),
                   " &nbsp; P(not_good) = ", round(prob[1, "not_good"], 3), "</p>"))
    }
  })
}

shinyApp(ui, server)
