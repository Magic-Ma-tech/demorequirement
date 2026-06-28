import os
import shutil
import time
import traceback
import uuid
import secrets
import threading
from pathlib import Path

import gradio as gr

from attack_core import surrogate_attack


BASE_DIR = Path(__file__).resolve().parent

APP_PREFIX = "google"

UPLOAD_DIR = BASE_DIR / f"{APP_PREFIX}_saved_uploads"
RESULT_DIR = BASE_DIR / f"{APP_PREFIX}_saved_results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# User configuration
# =========================
# For real deployment, do not store plain-text passwords in code.
# Use environment variables, a database, or hashed passwords instead.
USERS = {
    "watermarktest": {
        "password": "watermark",
    },
}


# =========================
# Login attempt control
# =========================
MAX_LOGIN_ATTEMPTS = 10
LOCKOUT_SECONDS = 15 * 60

FAILED_LOGIN_ATTEMPTS = {}
LOCKED_UNTIL = {}

LOGIN_LOCK = threading.Lock()


def authenticate(username: str, password: str) -> bool:
    """
    Validate Gradio login credentials.

    Each username is allowed a maximum of 10 failed login attempts.
    After that, the username is locked for 15 minutes.
    """
    username = username.strip() if username else ""
    password = password or ""

    now = time.time()

    with LOGIN_LOCK:
        locked_until = LOCKED_UNTIL.get(username)

        if locked_until is not None:
            if now < locked_until:
                return False

            LOCKED_UNTIL.pop(username, None)
            FAILED_LOGIN_ATTEMPTS.pop(username, None)

        current_failed_attempts = FAILED_LOGIN_ATTEMPTS.get(username, 0)

        user = USERS.get(username)

        if user is None:
            current_failed_attempts += 1
            FAILED_LOGIN_ATTEMPTS[username] = current_failed_attempts

            if current_failed_attempts >= MAX_LOGIN_ATTEMPTS:
                LOCKED_UNTIL[username] = now + LOCKOUT_SECONDS

            return False

        is_password_correct = secrets.compare_digest(password, user["password"])

        if is_password_correct:
            FAILED_LOGIN_ATTEMPTS.pop(username, None)
            LOCKED_UNTIL.pop(username, None)
            return True

        current_failed_attempts += 1
        FAILED_LOGIN_ATTEMPTS[username] = current_failed_attempts

        if current_failed_attempts >= MAX_LOGIN_ATTEMPTS:
            LOCKED_UNTIL[username] = now + LOCKOUT_SECONDS

        return False


def save_uploaded_audio(uploaded_audio_path: str) -> str:
    """
    Save the uploaded audio file from Gradio's temporary path
    to a persistent local directory.
    """
    source_path = Path(uploaded_audio_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded file does not exist: {uploaded_audio_path}")

    file_suffix = source_path.suffix or ".wav"
    saved_filename = f"{APP_PREFIX}_uploaded_{uuid.uuid4().hex}{file_suffix}"
    saved_path = UPLOAD_DIR / saved_filename

    shutil.copy2(source_path, saved_path)

    return str(saved_path)


def run_attack(uploaded_audio_path, request: gr.Request):
    """
    Run the audio timbre attack feature.

    When using gr.Audio(type='filepath'), Gradio passes the temporary
    server-side audio file path to this function.
    """
    username = request.username

    if uploaded_audio_path is None:
        raise gr.Error("Please upload an audio file first.")

    try:
        saved_upload_path = save_uploaded_audio(uploaded_audio_path)

        save_dir = RESULT_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        output_path, info = surrogate_attack(
            wav_path=saved_upload_path,
            save_dir=str(save_dir),
        )

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file does not exist: {output_path}")

        status_info = (
            f"Processing completed successfully.\n\n"
            f"Current user:\n{username}\n\n"
            f"Surrogate model attack info:\n{info}"
        )

        return output_path, output_path, status_info

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        raise gr.Error(f"Processing failed: {str(e)}")


with gr.Blocks(title="Audio Timbre Attack Demo") as demo:
    gr.Markdown("# Audio Timbre Attack Demo")
    gr.Markdown(
        "Upload an audio file. After backend processing is complete, "
        "a playable and downloadable `.wav` file will be returned."
    )

    input_audio = gr.Audio(
        label="Upload Audio",
        type="filepath",
        sources=["upload"],
    )

    run_button = gr.Button("Start Processing", variant="primary")

    output_audio = gr.Audio(
        label="Processed Audio",
        type="filepath",
    )

    output_file = gr.File(
        label="Download Processed WAV File",
    )

    status_box = gr.Textbox(
        label="Status",
        lines=6,
        interactive=False,
    )

    run_button.click(
        fn=run_attack,
        inputs=input_audio,
        outputs=[output_audio, output_file, status_box],
    )


if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=2,
    ).launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        max_file_size="50mb",
        show_error=True,
        auth=authenticate,
        auth_message="Please enter your username and password to log in.",
    )