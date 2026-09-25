# Data source note

App Store (iOS) scraping was attempted via app-store-scraper but consistently
failed with a JSON parsing error (Apple's undocumented review API appears to
have changed/broken independent of network access - confirmed reachable via
direct HTTP request, status 200). Proceeding with Google Play as the sole
data source: 83,150 reviews across Tim Hortons, Air Canada, and RBC, with
separate en/fr locale queries giving a genuine bilingual split. iOS noted as
a potential future extension if the library is patched or an alternative
scraping method is found.
