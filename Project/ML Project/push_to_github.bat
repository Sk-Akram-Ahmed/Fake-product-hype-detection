@echo off
REM GitHub Repository Creation and Push Script for Windows

echo === FAKE PRODUCT HYPE DETECTION - GITHUB SETUP ===
echo.

REM Step 1: Instructions for creating repository
echo 📋 STEP 1: Create GitHub Repository
echo 1. Go to https://github.com and log in
echo 2. Click the '+' button in the top right corner
echo 3. Select 'New repository'
echo 4. Fill in the repository details:
echo    - Repository name: fake-product-hype-detection
echo    - Description: Multimodal AI system for detecting artificially inflated product popularity
echo    - Visibility: Choose Public or Private
echo    - DO NOT initialize with README, .gitignore, or license
echo 5. Click 'Create repository'
echo.

REM Step 2: Get username
set /p username="📋 STEP 2: Enter Your GitHub Username: "

if "%username%"=="" (
    echo ❌ Username cannot be empty. Exiting.
    pause
    exit /b 1
)

REM Step 3: Add remote and push
echo.
echo 📋 STEP 3: Connect and Push to GitHub
echo Adding remote origin...
git remote add hype-origin https://github.com/%username%/fake-product-hype-detection.git

echo Pushing to GitHub...
git push -u hype-origin main

echo.
echo ✅ SUCCESS! Your code has been pushed to GitHub!
echo 🔗 Repository URL: https://github.com/%username%/fake-product-hype-detection
echo.
echo 📋 Next Steps:
echo 1. Visit your repository to verify all files are uploaded
echo 2. Set up GitHub OAuth following the Authentication Setup in README.md
echo 3. Test the application locally: streamlit run app.py
echo 4. Consider deploying to Streamlit Cloud or your preferred platform
echo.
pause
