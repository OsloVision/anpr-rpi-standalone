# API Key Setup for ANPR System

## 🔑 Required API Keys

To use the ANPR (Automatic Number Plate Recognition) postprocessing functionality, you need to configure API keys in the `.env` file.

### 1. OpenAI API Key (Required)
- **Purpose**: License plate text extraction using GPT-4o-mini
- **How to get**: 
  1. Go to https://platform.openai.com/api-keys
  2. Sign up/log in to your OpenAI account
  3. Create a new API key
  4. Copy the key (starts with `sk-`)

### 2. Norwegian Vehicle API Key (Optional)
- **Purpose**: Official Norwegian vehicle registry lookup
- **Fallback**: If not provided, the system will use web scraping as backup
- **How to get**: Contact Norwegian Public Roads Administration for API access

## 📝 Setup Instructions

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file:**
   ```bash
   nano .env
   ```

3. **Add your actual API keys:**
   ```bash
   # Replace with your actual OpenAI API key
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   
   # Optional: Add Norwegian Vehicle API key
   VEHICLE_API_KEY=your-actual-vehicle-api-key-here
   ```

4. **Save and restart the application**

## 🔒 Security Notes

- ⚠️ **Never commit the `.env` file to git** - it contains sensitive API keys
- ✅ The `.env` file is already in `.gitignore` for protection
- 🔄 Use `.env.example` as a template for sharing configuration structure
- 🔐 Keep your API keys secure and don't share them publicly

## 🚀 Testing Configuration

After setting up your API keys, the Streamlit UI will show:
- ✅ Green checkmark for properly configured keys
- ❌ Red error for missing/invalid keys
- ⚠️ Yellow warning for optional keys

The ANPR postprocessing button will be:
- **Enabled** when OpenAI API key is configured
- **Disabled** when OpenAI API key is missing

## 💡 API Key Status

The application automatically checks your API key configuration and provides helpful feedback in the UI. Look for the "🔑 API Configuration" section in Step 3 (Postprocess).

## 📞 Support

- **OpenAI API Issues**: Check https://platform.openai.com/docs
- **Norwegian Vehicle API**: Contact Norwegian Public Roads Administration
- **General Setup**: Check that `.env` file is in the correct directory