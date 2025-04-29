library(dplyr)
library(tidyr)
library(lubridate)
source("lunch.R")

kiosk2025 <- read.csv("~/deepL-cafeteria-main/data/ASIJ Lunchbox Sales History Feb-Apr 2025 - Feb-Apr 2025.csv")

kiosk2025 <- kiosk2025 %>% mutate(
  TransactionDateTimeUTC = ymd_hms(TransactionDateTimeUTC),
  JST.Transaction.Time = with_tz(kiosk2025$TransactionDateTimeUTC, tzone = "Asia/Tokyo"),
  
  #gets rid of grade prefix (0-12) *0 = kindergarten
  Grade = ifelse(Grade == "Kindergarten", "Grade 0", Grade),
  Grade = ifelse(grepl("Grade", Grade), as.integer(sub("Grade ", "", Grade)), NA_character_),
  
  TransactionAmount = -1 * TransactionAmount
  
)

kiosk2025$DayOfWeek <- wday(as.Date(kiosk2025$JST.Transaction.Time),label = TRUE)

#moving day of week next to the time
kiosk2025 <- kiosk2025[, c(1, 2, ncol(kiosk2025), 3:7)]

product_rank <- kiosk2025 %>% count(ProductName, sort = TRUE) %>% mutate(rank = row_number())

date_sort <- kiosk2025 %>% arrange(JST.Transaction.Time)

kiosk2025$date <- ymd(as.Date(kiosk2025$JST.Transaction.Time))



lunch2025 <- kiosk2025 %>% filter(TransactionAmount == 800)
lunch_ops <- unique(lunch2025$ProductName)


lunch_list <- unique(lunch2025$ProductName)
lunch_df <- df %>%
  filter(ProductName %in% lunch_ops)

daily_sort_before <- date_comparison(df)
daily_sort_after <- date_comparison(lunch2025)

daily_sort_combined <- rbind(daily_sort_before, daily_sort_after)
write.csv(daily_sort_combined, "daily_sort.csv")


unique(df$ProductName)
view(df$ProductName)
