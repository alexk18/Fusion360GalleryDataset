"""
Fusion 360 Gym — пошаговое построение стула (сиденье, ножки, спинка, подлокотники).
Запуск: Fusion 360 открыт, аддон Fusion 360 Gym запущен.
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
DELAY_SEC = 1.2


def step(client, msg, fn, *args, **kwargs):
    print(msg)
    r = fn(*args, **kwargs) if kwargs else fn(*args)
    time.sleep(DELAY_SEC)
    return r


def rect_points(cx, cy, w, h):
    """Углы прямоугольника (половины ширины/высоты)."""
    hw, hh = w / 2, h / 2
    return [
        {"x": cx - hw, "y": cy - hh},
        {"x": cx + hw, "y": cy - hh},
        {"x": cx + hw, "y": cy + hh},
        {"x": cx - hw, "y": cy + hh},
    ]


def draw_rect(client, sketch_name, pts):
    for p in pts:
        step(client, f"    точка ({p['x']:.1f}, {p['y']:.1f})", client.add_point, sketch_name, p)
    r = step(client, "    замыкание контура", client.close_profile, sketch_name)
    return r


def main():
    print("Подключение к Fusion 360 Gym...")
    client = Fusion360GymClient(f"http://{HOST_NAME}:{PORT_NUMBER}")
    step(client, "Очистка документа", client.clear)

    # --- 1. Сиденье (XY, 50x50 см, толщина 5 см) ---
    print("\n--- 1. Сиденье ---")
    r = step(client, "  Скетч на плоскости XY", client.add_sketch, "XY")
    if r.status_code != 200:
        print("Ошибка:", r.json().get("message", r.text))
        return
    sketch_seat = r.json()["data"]["sketch_name"]
    r = draw_rect(client, sketch_seat, rect_points(0, 0, 50, 50))
    if r.status_code != 200 or "profiles" not in r.json().get("data", {}):
        return
    step(client, "  Выдавливание сиденья 5 см", client.add_extrude,
         sketch_seat, next(iter(r.json()["data"]["profiles"])), 5, "NewBodyFeatureOperation")
    step(client, "  Подгонка вида под модель", client.refresh)

    # --- 2. Четыре ножки (на XY, по углам, выдавлены вниз) ---
    print("\n--- 2. Ножки ---")
    r = step(client, "  Скетч для ножек (XY)", client.add_sketch, "XY")
    if r.status_code != 200:
        return
    sketch_legs = r.json()["data"]["sketch_name"]
    leg_size = 5
    leg_positions = [(22.5, 22.5), (22.5, -22.5), (-22.5, 22.5), (-22.5, -22.5)]
    profile_ids = []
    for i, (cx, cy) in enumerate(leg_positions):
        r = draw_rect(client, sketch_legs, rect_points(cx, cy, leg_size, leg_size))
        if r.status_code == 200 and "profiles" in r.json().get("data", {}):
            profile_ids.extend(r.json()["data"]["profiles"].keys())
    for i, pid in enumerate(profile_ids[:4]):
        step(client, f"  Выдавливание ножки {i+1} вниз 12 см", client.add_extrude,
             sketch_legs, pid, -12, "NewBodyFeatureOperation")
    step(client, "  Подгонка вида под модель", client.refresh)

    # --- 3. Спинка (YZ, позади сиденья, выдавить в −X) ---
    print("\n--- 3. Спинка ---")
    r = step(client, "  Скетч на плоскости YZ", client.add_sketch, "YZ")
    if r.status_code != 200:
        return
    sketch_back = r.json()["data"]["sketch_name"]
    # В YZ: y — высота (от 5 до 40), z — ширина (-25..25)
    r = draw_rect(client, sketch_back, rect_points(22.5, 0, 35, 50))
    if r.status_code != 200 or "profiles" not in r.json().get("data", {}):
        return
    pid = next(iter(r.json()["data"]["profiles"]))
    step(client, "  Выдавливание спинки 25 см в −X", client.add_extrude,
         sketch_back, pid, -25, "NewBodyFeatureOperation")
    step(client, "  Подгонка вида", client.refresh)

    # --- 4. Пять вертикальных планок спинки (упрощённо: 5 тонких тел на YZ) ---
    print("\n--- 4. Планки спинки ---")
    r = step(client, "  Скетч для планок (YZ)", client.add_sketch, "YZ")
    if r.status_code != 200:
        return
    sketch_slats = r.json()["data"]["sketch_name"]
    slat_w = 3
    z_centers = [-20, -10, 0, 10, 20]
    for i, zc in enumerate(z_centers):
        r = draw_rect(client, sketch_slats, rect_points(22.5, zc, 35, slat_w))
        if r.status_code == 200 and "profiles" in r.json().get("data", {}):
            pid = next(iter(r.json()["data"]["profiles"]))
            step(client, f"  Выдавливание планки {i+1}", client.add_extrude,
                 sketch_slats, pid, -25, "NewBodyFeatureOperation")

    # --- 5. Левый подлокотник (YZ, слева, выдавлен в −X) ---
    print("\n--- 5. Левый подлокотник ---")
    r = step(client, "  Скетч для левого подлокотника (YZ)", client.add_sketch, "YZ")
    if r.status_code != 200:
        return
    sketch_left = r.json()["data"]["sketch_name"]
    r = draw_rect(client, sketch_left, rect_points(22.5, -15, 20, 5))
    if r.status_code != 200 or "profiles" not in r.json().get("data", {}):
        return
    pid = next(iter(r.json()["data"]["profiles"]))
    step(client, "  Выдавливание левого подлокотника", client.add_extrude,
         sketch_left, pid, -20, "NewBodyFeatureOperation")

    # --- 6. Правый подлокотник ---
    print("\n--- 6. Правый подлокотник ---")
    r = step(client, "  Скетч для правого подлокотника (YZ)", client.add_sketch, "YZ")
    if r.status_code != 200:
        return
    sketch_right = r.json()["data"]["sketch_name"]
    r = draw_rect(client, sketch_right, rect_points(22.5, 15, 20, 5))
    if r.status_code != 200 or "profiles" not in r.json().get("data", {}):
        return
    pid = next(iter(r.json()["data"]["profiles"]))
    step(client, "  Выдавливание правого подлокотника", client.add_extrude,
         sketch_right, pid, -20, "NewBodyFeatureOperation")

    step(client, "Обновление вида и подгонка камеры под весь стул", client.refresh)
    print("\nСтул готов. Проверьте вид в Fusion 360.")


if __name__ == "__main__":
    main()
