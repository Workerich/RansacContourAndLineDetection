import os
import uuid
import shutil
import time
import ransac as rc
import mnk
from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

BASE_DIR       = os.path.dirname(__file__)
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
MAX_DIMENSION  = 1000   # px
MAX_PIXELS = 1000000  # px^2
SESSION_TTL    = 3600   # секунд до автоудаления (1 час)
# Сколько секунд между проверками GC при каждом входящем запросе
GC_INTERVAL    = 300    # 5 минут

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# { session_id: last_seen_timestamp }
# Хранится в памяти процесса; на PythonAnywhere один воркер
_sessions: dict = {}
_last_gc: float = 0.0


def _touch(session_id: str) -> None:
    _sessions[session_id] = time.time()


def _remove_session(session_id: str) -> None:
    _sessions.pop(session_id, None)

    # Входные файлы: uploads/<session_id>.<ext>
    for name in os.listdir(UPLOAD_FOLDER):
        if name.startswith(session_id):
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, name))
            except OSError:
                pass

    # Папка результатов: results/<session_id>/
    result_dir = os.path.join(RESULTS_FOLDER, session_id)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)


def _gc_tick() -> None:
    """
    Вызывается при каждом запросе, но реально работает не чаще раз в GC_INTERVAL.
    Потоки не нужны — PythonAnywhere их не поддерживает.
    """
    global _last_gc
    now = time.time()
    if now - _last_gc < GC_INTERVAL:
        return
    _last_gc = now

    stale = [sid for sid, ts in list(_sessions.items()) if now - ts > SESSION_TTL]
    for sid in stale:
        _remove_session(sid)


# ── Lazy-import helpers ───────────────────────────────────────────────────────
# cv2 и PIL импортируются только там, где реально нужны,
# чтобы не тратить CPU на холодный старт и на лёгкие маршруты.

def _open_image_pil(filepath: str):
    from PIL import Image
    with Image.open(filepath) as img:
        img.verify()


def _check_image_size(filepath: str):
    """Возвращает (width, height) через cv2. Импорт только здесь."""
    import cv2
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError('OpenCV не смог открыть файл')
    h, w = img.shape[:2]
    return w, h


def run_contour_script(input_path, session_id, w, E, min_length, max_lines, filtered):
    """
    Здесь будет выполнен скрипт обнаружения контуров.

    Параметры:
        input_path  – путь до входного изображения
        session_id  – уникальный ID сессии
        w           – доля «хороших» точек (0 < w < 1)
        E           – порог функции оценки принадлежности
        min_length  – минимальная длина линии, пиксели
        max_lines   – максимальное количество линий
        filtered    – фильтровать выбросы на каждой итерации (bool)

    Должен сохранить:
        results/<session_id>/ransac.png
        results/<session_id>/mnk.png

    И вернуть словарь с ключами text1, text2, ransac_img, mnk_img.
    """
    result_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(result_dir, exist_ok=True)

    import cv2
    msg_rc, img = rc.process_ransac(input_path, w, E, min_length, max_lines, filtered)
    cv2.imwrite(os.path.join(result_dir, 'ransac.png'), img)

    msg_mnk, img = mnk.process_mnk(input_path)
    cv2.imwrite(os.path.join(result_dir, 'mnk.png'), img)

    return {
        'text1':      msg_rc,
        'text2':      msg_mnk,
        'ransac_img': f'/results/{session_id}/ransac.png',
        'mnk_img':    f'/results/{session_id}/mnk.png',
    }

# Routes

@app.before_request
def before_request():
    _gc_tick()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден в запросе'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Файл не выбран'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Недопустимый тип файла. Разрешены: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF'}), 400

    session_id = str(uuid.uuid4())
    filename   = f'{session_id}.{ext}'
    filepath   = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        _open_image_pil(filepath)
    except Exception:
        os.remove(filepath)
        return jsonify({'error': 'Файл повреждён или не является изображением'}), 400

    _touch(session_id)
    return jsonify({'session_id': session_id, 'filename': filename})


@app.route('/detect', methods=['POST'])
def detect():
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    filename   = data.get('filename', '')

    if not session_id or not filename:
        return jsonify({'error': 'Неверные параметры запроса'}), 400

    try:
        w          = float(data['w'])
        E          = float(data['E'])
        min_length = int(data['min_length'])
        max_lines  = int(data['max_lines'])
        filtered   = bool(data.get('filtered', False))
    except (KeyError, ValueError, TypeError) as ex:
        return jsonify({'error': f'Ошибка параметров: {ex}'}), 400

    errors = []
    if not (0 < w < 1.0):
        errors.append('w должно быть строго между 0 и 1')
    if E <= 0:
        errors.append('E должно быть больше 0')
    if min_length <= 0:
        errors.append('min_length должно быть больше 0')
    if max_lines <= 0:
        errors.append('max_lines должно быть больше 0')
    if errors:
        return jsonify({'error': '; '.join(errors)}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден на сервере'}), 404

    try:
        img_w, img_h = _check_image_size(filepath)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    print(img_w, img_h)
    if img_w * img_h > MAX_PIXELS:
        return jsonify({
            'error':      (f'Допустимый размер — менее {MAX_DIMENSION}×{MAX_DIMENSION} пикселей. '
                           f'Ваше изображение: {img_w}×{img_h} px.'),
            'size_error': True,
        }), 422

    try:
        results = run_contour_script(
            filepath, session_id, w, E, min_length, max_lines, filtered
        )
        _touch(session_id)
    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке: {e}'}), 500
    finally:
        # Входное изображение удаляем сразу после обработки
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

    return jsonify(results)


@app.route('/results/<session_id>/<filename>')
def serve_result(session_id, filename):
    # только alphanum + дефис в session_id
    if not all(c.isalnum() or c == '-' for c in session_id):
        return jsonify({'error': 'bad session'}), 400
    _touch(session_id)
    result_dir = os.path.join(RESULTS_FOLDER, session_id)
    return send_from_directory(result_dir, filename)


@app.route('/ping', methods=['POST'])
def ping():
    """Heartbeat от клиента — продлевает TTL сессии."""
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    if session_id:
        _touch(session_id)
    return '', 204  # No Content — минимальный ответ


@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Явное удаление сессии (вызывается при закрытии вкладки)."""
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    if session_id:
        _remove_session(session_id)
    return '', 204

if __name__ == "__main__":
    app.run(debug=True, port=5000)