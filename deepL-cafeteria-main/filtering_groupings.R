library(dplyr)
library(tidyr)
library(lubridate)
source("lunch.R")

kiosk2025 <- read.csv("~/deepL-cafeteria-main/data/ASIJ Lunchbox Sales History Feb-Apr 2025 - Feb-Apr 2025.csv")

kiosk2025 <- kiosk2025 %>% mutate(
  TransactionDateTimeUTC = ymd_hms(TransactionDateTimeUTC),
  JST.Transaction.Time = ymd_hms(JST.Transaction.Time),
  
  #gets rid of grade prefix (0-12) *0 = kindergarten
  Grade = ifelse(Grade == "Kindergarten", "Grade 0", Grade),
  Grade = ifelse(grepl("Grade", Grade), as.integer(sub("Grade ", "", Grade)), NA_character_),
  
  TransactionAmount = -1 * TransactionAmount
  
)



lunch2025 <- kiosk2025 %>% filter(TransactionAmount == 800)
lunch_ops <- unique(lunch2025$ProductName)

lunch_df <- data.frame()

for (i in 1:nrow(df)){
  for (j in 1:nrow(lunch_ops)){
    
  }
}



unique(df$ProductName)
view(df$ProductName)
