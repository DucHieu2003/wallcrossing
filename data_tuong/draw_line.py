import argparse
import json
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

window_name = "Draw Wall"

walls = []           # finished walls: list of polylines, each polyline = list of (x, y)
current_points = []  # points of the wall being drawn
image = None
original_image = None
current_image_path = None


def redraw():
    global image
    image = original_image.copy()

    info = f"{current_image_path.name} | walls: {len(walls)} | drawing: {len(current_points)} pts"
    help_text = "left: add point | f/Enter: finish wall | z/right: undo | s: save+next | n: skip | q: quit"
    cv2.putText(image, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, help_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for wall in walls:
        draw_polyline(wall, (0, 255, 0))

    if current_points:
        draw_polyline(current_points, (255, 0, 0))

    cv2.imshow(window_name, image)


def draw_polyline(points, color):
    for i, point in enumerate(points):
        cv2.circle(image, point, 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(image, points[i - 1], point, color, 2)


def finish_wall():
    if len(current_points) >= 2:
        walls.append(list(current_points))
        print(f"Wall {len(walls)}: {len(current_points)} points")
        current_points.clear()
        redraw()
    elif current_points:
        print("Need at least 2 points to finish a wall.")


def undo():
    if current_points:
        current_points.pop()
        print("Removed last point.")
    elif walls:
        removed = walls.pop()
        print(f"Removed last wall ({len(removed)} points).")
    redraw()


def handle_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        undo()


def find_images(input_path):
    if input_path.is_file():
        return [input_path]

    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_walls_data():
    return [
        [{"x": p[0], "y": p[1]} for p in wall]
        for wall in walls
    ]


def save_walls(image_path):
    output_path = image_path.with_suffix(".json")
    data = {"image": image_path.name, "walls": make_walls_data()}
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


def load_image(image_path):
    global original_image, image, current_image_path

    current_image_path = image_path
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Cannot read image, skipped: {image_path}")
        return False

    image = original_image.copy()
    walls.clear()
    current_points.clear()
    redraw()
    return True


def main():
    parser = argparse.ArgumentParser(description="Draw wall polylines on many images and save to JSON.")
    parser.add_argument("input_path", help="Image file or folder containing images")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    image_paths = find_images(input_path)

    if not image_paths:
        raise SystemExit(f"No images found: {input_path}")

    print(f"Found {len(image_paths)} images.")
    print("Left click: add point to current wall (draw curved/long walls with many points)")
    print("Press f or Enter: finish current wall, start a new one")
    print("Press z or right click: undo last point / last wall")
    print("Press s: save all walls and next image")
    print("Press n: skip image without wall")
    print("Press q or Esc: save and quit")

    index = 0

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse)

    while index < len(image_paths):
        if not load_image(image_paths[index]):
            index += 1
            continue

        image_path = image_paths[index]

        moved = False
        while not moved:
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("f"), 13):  # finish current wall
                finish_wall()

            elif key == ord("z"):
                undo()

            elif key == ord("s"):
                finish_wall()  # auto-finish any in-progress wall
                save_walls(image_path)
                print(f"Done {index + 1}/{len(image_paths)} ({len(walls)} walls): {image_path.name}")
                index += 1
                moved = True

            elif key == ord("n"):
                save_walls(image_path)
                print(f"Skipped {index + 1}/{len(image_paths)}: {image_path.name}")
                index += 1
                moved = True

            elif key in (ord("q"), 27):
                finish_wall()
                save_walls(image_path)
                cv2.destroyAllWindows()
                print("Finished.")
                return

    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()
