# 📊 CMT Comparable Transactions Finder

This Streamlit app helps financial analysts and valuation professionals identify **comparable M&A transactions** based on semantic similarity to a provided company profile.

---

## 🚀 Features

- 🔍 Scrapes websites for company information.
- 🤖 Uses OpenAI's `text-embedding-ada-002` to compute semantic embeddings.
- 🧠 Ranks and displays the most relevant comparable transactions.
- 📥 Downloads results to Excel.
- ✅ Optimized for repeated usage via embedding and scraping cache.

---

## 📁 File Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml     # Streamlit UI configuration
└── app_data/
    └── Database.xlsx          # M&A transactions database
```

---

## 🔧 Setup and Deployment

### Option 1: Run Locally

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Set OpenAI API Key**:
    - Create a `.streamlit/secrets.toml` file:
      ```toml
      [openai]
      api_key = "your-openai-api-key"
      ```

3. **Run the app**:
    ```bash
    streamlit run app.py
    ```

---

### Option 2: Deploy to Streamlit Cloud

1. **Push this code to a GitHub repo**.
2. **Create a new app** on [Streamlit Cloud](https://share.streamlit.io).
3. **Configure secrets** under the `Secrets` tab:
    ```toml
    [openai]
    api_key = "your-openai-api-key"
    ```

4. **Deploy and use the app** via browser!

---

## 🧠 How It Works

- M&A database entries are enriched using scraped website content.
- A combined text (`Business Description + Industry + Scraped Text`) is embedded via OpenAI API.
- Cosine similarity is calculated between the input profile and all records.
- Top matches are shown and downloadable.

---

## 📬 Contact

Developed by **CMT Advisory**. For questions or support, contact: `zamowienia@cmt-advisory.pl`