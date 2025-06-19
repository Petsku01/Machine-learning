# Phishing Detection Script README

This repository contains a series of Python scripts (`phishing_detection_iterX.py`, where X ranges from 1 to 9) that implement a phishing URL detection model using logistic regression, URL feature extraction, and TF-IDF vectorization. The scripts evolve through nine iterations, with each version introducing new features, fixing bugs, or improving robustness. Below is a detailed summary of updates and fixes for each iteration.

## Overview
The scripts aim to classify URLs as legitimate (0) or phishing (1) based on features like URL length, digit count, special character count, HTTPS usage, subdomain count, and TF-IDF text features. The dataset starts small (4–6 URLs) and is intended for demo purposes, with warnings about its limitations. The final iteration (9) is the most robust but remains a demo, not production-ready.

## Dependencies
- Python 3.6+
- Libraries: `pandas`, `numpy`, `scikit-learn` (all iterations except Iteration 1, which only needs `pandas` and `scikit-learn`)
- Installation: `pip install pandas numpy scikit-learn`

## Testing 
1. Save each script as `phishing_detection_iterX.py`.
2. Install dependencies.
3. Run: `python phishing_detection_iterX.py`.
4. Check console output for accuracy, classification reports, and predictions.
5. Note: Earlier iterations may crash on invalid inputs due to limited error handling.

## Iteration Summaries

### Iteration 1 (`phishing_detection_iter1.py`)
- **Features**:
  - Basic URL feature extraction: URL length, HTTPS presence (`has_secure`).
  - Logistic regression model with accuracy evaluation.
  - Small dataset (4 URLs: 2 legitimate, 2 phishing).
  - 50% test split (`test_size=0.5`).
- **Issues**:
  - No input validation for URLs.
  - No error handling for `urlparse` failures.
  - Limited features, no TF-IDF vectorization.
  - No classification report, only accuracy.
  - No prediction for new URLs.
  - Small test set risks unreliable results.

### Iteration 2 (`phishing_detection_iter2.py`)
- **Updates/Fixes**:
  - Added features: `num_digits` (count of digits in URL), `num_special_chars` (count of special characters using regex `[!@#$%^&*]`).
  - Added `numpy` for numerical operations.
  - Added `classification_report` for detailed evaluation.
- **Issues**:
  - No input validation or error handling.
  - Limited special characters regex misses many characters (e.g., commas, periods).
  - Case-sensitive HTTPS check (`has_secure` fails on `HTTPS`).
  - No TF-IDF features.
  - No `zero_division` parameter in `classification_report`, risking errors.
  - No new URL prediction.

### Iteration 3 (`phishing_detection_iter3.py`)
- **Updates/Fixes**:
  - Introduced TF-IDF vectorization for URL text features using `TfidfVectorizer`.
  - Added basic input validation (checks if URL is a string).
  - Expanded dataset to 6 URLs (3 legitimate, 3 phishing).
  - Improved special characters regex to include more characters (`[!@#$%^&*(),.?]`).
  - Stored original URL in dataset for TF-IDF (`url` column).
- **Issues**:
  - No regex for allowed characters in URLs.
  - Case-sensitive HTTPS check persists.
  - TF-IDF lacks parameters (e.g., `analyzer`, `ngram_range`).
  - No column names for TF-IDF features, risking misalignment.
  - No test set size check or empty dataset handling.
  - No new URL prediction.

### Iteration 4 (`phishing_detection_iter4.py`)
- **Updates/Fixes**:
  - Added `subdomains` feature (counts subdomains in netloc).
  - Added regex validation for URLs (`^[a-zA-Z0-9:/?=&._@%#-+~]+$`).
  - Added scheme validation (requires scheme to be present).
  - Introduced new URL prediction with a sample URL.
  - Added warnings for small dataset and demo nature.
  - Added `zero_division=0` in `classification_report`.
- **Bugs Introduced**:
  - Typo: `feature['subdomains']` instead of `features['subdomains']`.
  - Incorrect feature selection: `num_subdomains` instead of `subdomains`.
  - Incorrect `TfidfVectorizer` parameters: `max_analyzer`, `errors=ngram_range`.
  - Incorrect method: `get_features_names_out` instead of `get_feature_names_out`.
  - Prediction bug: Uses `predictions[0]` instead of `prediction[0]`.

### Iteration 5 (`phishing_detection_iter5.py`)
- **Updates/Fixes**:
  - Fixed typos: Corrected `feature` to `features`, `num_subdomains` to `subdomains`.
  - Fixed `TfidfVectorizer` parameters: Set `analyzer='char'`, `ngram_range=(1,2)`, `max_features=20`.
  - Fixed method: Corrected `get_features_names_out` to `get_feature_names_out`.
  - Fixed prediction bug: Used `prediction[0]` correctly.
  - Renamed `has_secure` to `has_https` and made it case-insensitive (`parsed.scheme.lower()`).
  - Added robust scheme validation (only `http` or `https` allowed).
  - Added netloc validation to reject complex formats (e.g., IPv6, excluding ports).
  - Added test set size check (`len(X_test) < 2`).
  - Added warning for small test set size.
  - Allowed brackets (`[]`) in URL regex.
- **Issues**:
  - No dependency checks.
  - No URL length limits.
  - No single quotes or parentheses in regex.
  - No URL decoding for percent-encoded characters.

### Iteration 6 (`phishing_detection_iter6.py`)
- **Updates/Fixes**:
  - Updated regex to allow parentheses (`()`) in URLs (`^[a-zA-Z0-9:/?=&._@%#-+~\[\]\(\)]+$`).
- **Issues**:
  - Same as Iteration 5, except parentheses are now allowed.

### Iteration 7 (`phishing_detection_iter7.py`)
- **Updates/Fixes**:
  - Added dependency check to ensure `pandas`, `numpy`, and `scikit-learn` are installed.
  - Added URL length validation: Minimum 10 characters, maximum 2048.
  - Updated regex to allow single quotes (`'`): `^[a-zA-Z0-9:/?=&._@%#-+~\[\]\(\)\']+$`.
  - Aligned special characters regex to include single quotes, brackets, and parentheses: `[!@#$%^&*(),.?:;=<>~+#-\'\[\]\(\)]`.
  - Increased `LogisticRegression` iterations: `max_iter=1000` to ensure convergence.
- **Issues**:
  - No URL decoding for percent-encoded characters.
  - Warning message for test set size could be more precise.

### Iteration 8 (`phishing_detection_iter8.py`)
- **Updates/Fixes**:
  - Added URL decoding with `urllib.parse.unquote` to handle percent-encoded characters.
  - Updated warning message for test set size: Specifies at least 10 samples recommended.
  - Simplified special characters regex for clarity (no functional change).
- **Issues**:
  - Decoded URLs not re-validated, risking invalid characters (e.g., spaces from `%20`).

### Iteration 9 (`phishing_detection_iter9.py`)
- **Updates/Fixes**:
  - Added regex validation after URL decoding to ensure decoded URLs only contain allowed characters.
  - Finalized script as robust for demo purposes.
- **Remaining Limitations** (Not Errors):
  - Small dataset (6 URLs) limits model accuracy.
  - Basic features; production needs advanced features (e.g., domain age, WHOIS data).
  - Logistic regression is basic; ensemble models could improve performance.
  - Subdomain counting may miscalculate for complex TLDs (e.g., `co.uk`).
  - Regex may reject rare valid URLs with exotic characters.

## Key Evolution
- **Iterations 1–2**: Basic model with minimal features and no validation.
- **Iterations 3–4**: Introduced TF-IDF and validation but with significant bugs.
- **Iterations 5–6**: Fixed bugs, added robust validation, and improved usability.
- **Iterations 7–9**: Enhanced robustness with dependency checks, URL decoding, and refined validation.

## Usage Notes
- **Demo Purpose**: All scripts are for demonstration, not production. Iteration 9 is the most reliable.
- **Expected Output**: Accuracy, classification report, and prediction for a sample URL (`http://example.com-secure-login-paypal.com`).
- **Edge Cases**: Test with invalid URLs (e.g., `ftp://example.com`), short URLs, or percent-encoded URLs (`http://example.com/%20`) to observe error handling.
- **Improvements**: For production, use a larger dataset, add advanced features, and consider ensemble models.

## Contact
For questions or suggestions, please reach out via the repository's issue tracker.

*Last Updated: June 19, 2025*