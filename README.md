# Project Setup Instructions

Follow the steps below to set up the project locally and run the application.

---

## Prerequisits

### 1. Modal API Token

- Go to [Modal.com](https://modal.com).
- Signup or Login to your account.
- Navigate to `Profile Settings` > `API Tokens` > `New Token`.
- Note down the **token-id** and **token-secret**, or copy the command provided and run it directly to generate your token.

### 2. Hugging Face API Token

- Go to [Hugging Face](https://huggingface.co).
- Navigate to `Profile` > `Access Tokens` > `Create new token`.
- Set `token-type` to `write` and create the token.

### 3. Add Secrets to Modal

- Go to the [Modal dashboard](https://modal.com/dashboard).
- Navigate to `Secrets` > `Create New Secret`.
- Set the secret type to `hugging face` and add the Hugging Face token.

  
## Copy the .env.template file to .env:
```bash
cp template.env main.env
```

---

##  🐳 Running via Docker

To run the project using Docker, follow these steps:
Populate the .env file with your specific environment variables using the setup_env.sh script:
```bash
./setup.sh
```

Build and start the Docker containers:

    docker-compose up --build

---

## Manual Installation




## Step 1: Create a Virtual Environment (venv)

Create a Python virtual environment for the project:

```bash
python -m venv venv
```

Activate the virtual environment:

On Windows:

```bash
.\venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

## Step 2: Install Dependencies Using pip

Install the required Python dependencies using pip:
```bash
pip install -r requirements.txt
```
## Install ffmpeg

You need to install ffmpeg for multimedia processing. Follow the installation instructions below depending on your operating system.

 macOS (Homebrew):
```bash
brew install ffmpeg
```
Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```
Windows: Download and install ffmpeg from ffmpeg.org and add the path to your system environment variables.

## Step 3: Create .env File from Template

Populate the .env file with your specific environment variables using the setup_env.sh script:
```bash
./setup.sh
```

## Manual Running:

## Start the Frontend UI

   Navigate to the product UI folder:
```bash
cd ./src/UI/product_UI
```

Install the necessary dependencies:
```bash
pnpm install
```

Start the frontend UI:

```bash
  pnpm run dev
```

## Start the Backend


To start the backend for product, run the following command:
```bash
python main_app.py --serve product
```


To start the backend for genAI, run the following command:
```bash
python main_app.py --serve genAI
```


To start the backend for modl, use the following command:
```bash
python main_app.py --serve model
```


To start the backend for all modules (including genAI, model, and any others), run this command:
```bash
python main_app.py --serve all
```





