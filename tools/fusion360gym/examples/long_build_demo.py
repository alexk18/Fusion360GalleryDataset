"""
Fusion 360 Gym — длинное демо: пошаговое построение нескольких деталей с паузами.
Запускайте при работающем сервере Fusion 360 Gym. В терминале будет виден прогресс;
между шагами пауза DELAY_SEC, чтобы было время заметить обновления в Fusion (если UI успевает).
"""

from pathlib import Path
import sys
import os
import time

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")
if CLIENT_DIR not in sys.path:
    sys.path.append(CLIENT_DIR)

from fusion360gym_client import Fusion360GymClient

HOST_NAME = "127.0.0.1"
PORT_NUMBER = 8080
# Пауза между шагами (секунды) — увеличьте, если хотите дольше смотреть каждый шаг
DELAY_SEC = 1.5


def step(client, msg, fn, *args, **kwargs):
    """Выполнить один шаг, вывести сообщение и сделать паузу."""
    print(msg)
    r = fn(*args, **kwargs) if kwargs else fn(*args)
    time.sleep(DELAY_SEC)
    return r


def add_square_extrude(client, sketch_plane, size, distance, operation="NewBodyFeatureOperation"):
    """Создать скетч-квадрат на плоскости и выдать выдавливание."""
    r = step(client, f"  Скетч на плоскости {sketch_plane}...", client.add_sketch, sketch_plane)
    if r.status_code != 200:
        return None
    sketch_name = r.json()["data"]["sketch_name"]
    half = size / 2
    pts = [
        {"x": -half, "y": -half},
        {"x": half, "y": -half},
        {"x": half, "y": half},
        {"x": -half, "y": half},
    ]
    for pt in pts:
        step(client, f"    Точка ({pt['x']}, {pt['y']})", client.add_point, sketch_name, pt)
    r = step(client, "  Замыкание контура...", client.close_profile, sketch_name)
    if r.status_code != 200:
        return None
    profile_id = next(iter(r.json()["data"]["profiles"]))
    step(client, f"  Выдавливание на {distance} см ({operation})...",
         client.add_extrude, sketch_name, profile_id, distance, operation)
    return r


def main():
    print("Подключение к Fusion 360 Gym...")
    client = Fusion360GymClient(f"http://{HOST_NAME}:{PORT_NUMBER}")
    step(client, "Очистка документа...", client.clear)

    # Блок 1: основание на XY
    print("\n--- Блок 1: Основание (XY) ---")
    add_square_extrude(client, "XY", 10, 2, "NewBodyFeatureOperation")

    # Блок 2: столбик на XZ
    print("\n--- Блок 2: Столбик (XZ) ---")
    add_square_extrude(client, "XZ", 4, 8, "NewBodyFeatureOperation")

    # Блок 3: ещё один параллелепипед на YZ
    print("\n--- Блок 3: Боковая часть (YZ) ---")
    add_square_extrude(client, "YZ", 3, 6, "NewBodyFeatureOperation")

    # Блок 4: маленький куб на XY (будет второе тело)
    print("\n--- Блок 4: Второе тело на XY ---")
    add_square_extrude(client, "XY", 2, 2, "NewBodyFeatureOperation")

    # Блок 5: ещё один куб
    print("\n--- Блок 5: Третье тело ---")
    add_square_extrude(client, "XZ", 1.5, 3, "NewBodyFeatureOperation")

    print("\nГотово. Проверьте вид в Fusion 360.")
    step(client, "Обновление вида...", client.refresh)


if __name__ == "__main__":
    main()
