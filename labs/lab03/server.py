from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

RESULTS_DIR = "/results"

def read_report(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        return "Ожидание завершения расчетов... (обновите страницу позже)"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Ошибка чтения файла: {e}"

@app.get("/", response_class=HTMLResponse)
async def get_results():
    report_01 = read_report("lab01_output.txt")
    report_02 = read_report("lab02_output.txt")

    html_content = f"""
    <html>
        <head>
            <meta http-equiv="refresh" content="3">
            <title>Вывод консолей лабораторных работ</title>
            <style>
                body {{ font-family: sans-serif; padding: 20px; background-color: #f0f2f5; }}
                .container {{ display: flex; gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; }}
                h1 {{ color: #333; }}
                h2 {{ color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 14px; }}
            </style>
        </head>
        <body>
            <h1>Вывод консолей лабораторных работ</h1>
            <div class="container">
                <div class="card">
                    <h2>Лабораторная №1 (Python vs NumPy)</h2>
                    <pre>{report_01}</pre>
                </div>
                <div class="card">
                    <h2>Лабораторная №2 (Numba vs PyBind11)</h2>
                    <pre>{report_02}</pre>
                </div>
            </div>
        </body>
    </html>
    """
    return html_content