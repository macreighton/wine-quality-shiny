## deploy.R
## One-time setup + deployment script for publishing this app to shinyapps.io
##
## 1. Create a free account at https://www.shinyapps.io
## 2. Go to Account -> Tokens -> Show/Add Token, copy the name/token/secret
## 3. Paste them below (or better: run setAccountInfo() once interactively
##    so the secret isn't stored in this file) and source this script.

if (!requireNamespace("rsconnect", quietly = TRUE)) {
  install.packages("rsconnect")
}
library(rsconnect)

## --- one-time account registration -----------------------------------
## Uncomment and fill in with YOUR credentials from shinyapps.io,
## then run once. After that you can delete/comment this block again.
# rsconnect::setAccountInfo(
#   name   = "YOUR-ACCOUNT-NAME",
#   token  = "YOUR-TOKEN",
#   secret = "YOUR-SECRET"
# )

## --- deploy -------------------------------------------------------------
rsconnect::deployApp(
  appDir       = ".",
  appName      = "wine-quality-explorer",
  appTitle     = "Wine Quality Explorer",
  account      = "YOUR-ACCOUNT-NAME",   # change me
  forceUpdate  = TRUE
)

## After deployment, your app will be live at:
##   https://YOUR-ACCOUNT-NAME.shinyapps.io/wine-quality-explorer/
