            image = cv2.rectangle(image, (int(x), int(y)),
                                  (int(x + w), int(y + h)), color_map[cat_id].tolist(), line_width)