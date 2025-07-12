@echo off
echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing required packages...
pip install --upgrade pip

:: Required libraries
pip install -r requirements.txt

:: Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

echo Setup completed. You can now run your Streamlit app.
pause
