library(dplyr)
library(tidyr)
library(lubridate)

df <- read.csv("~/deepL-cafeteria-main/data/ASIJ_sorted_cafeteria.csv")

df <- df %>% mutate(
  TransactionDateTimeUTC = ymd_hms(TransactionDateTimeUTC),
  JST.Transaction.Time = ymd_hms(JST.Transaction.Time),
  
  #gets rid of grade prefix (0-12) *0 = kindergarten
  Grade = ifelse(Grade == "Kindergarten", "Grade 0", Grade),
  Grade = ifelse(grepl("Grade", Grade), as.integer(sub("Grade ", "", Grade)), NA_character_),
  
  TransactionAmount = -1 * TransactionAmount
  
)

df$DayOfWeek <- wday(as.Date(df$JST.Transaction.Time),label = TRUE)

#moving day of week next to the time
df <- df[, c(1, 2, ncol(df), 3:7)]

product_rank <- df %>% count(ProductName, sort = TRUE) %>% mutate(rank = row_number())

date_sort <- df %>% arrange(JST.Transaction.Time)

df$date <- ymd(as.Date(df$JST.Transaction.Time))



head(df)

date_comparison <- function(x){
  df <- x %>%
    group_by(date, DayOfWeek, ProductName) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(date, desc(count))%>%
    group_by(date) %>%
    mutate(rank = row_number())%>%
    filter(rank <= 4) %>%
    mutate(total_count = sum(count))
  
  df_wide <- df %>%
    pivot_wider(names_from = rank, values_from = c(ProductName, count), 
                names_prefix = "")
  
  return(df_wide)
}

df_compare <- date_comparison(df)

write.csv(df_compare, "daily_sort.csv")

hs_df  <- df %>% filter(Grade >= 9)
ms_df  <- df %>% filter(Grade >= 6 & Grade <= 8)
es1_df <- df %>% filter(Grade < 4)
es2_df <- df %>% filter(Grade >= 4 & Grade <= 5)

hs_compare <- date_comparison(hs_df)
ms_compare <- date_comparison(ms_df)
es2_compare <- date_comparison(es2_df)
es1_compare <- date_comparison(es1_df)
