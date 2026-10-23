# -*- coding: utf-8 -*-
"""
Нейросеть для распознавания рукописных цифр (датасет MNIST).
Для запуска установите зависимости:
    pip install tensorflow numpy matplotlib
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. Загрузка датасета MNIST
    # ==========================================
    print("Загрузка данных MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"Обучающая выборка: {x_train.shape}")  # (60000, 28, 28)
    print(f"Тестовая выборка:  {x_test.shape}")   # (10000, 28, 28)

    # ==========================================
    # 2. Предобработка данных
    # ==========================================
    # Нормализация значений пикселей в диапазон [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Преобразование изображений 28x28 в одномерный вектор (784 признака)
    x_train = x_train.reshape(-1, 28 * 28)
    x_test  = x_test.reshape(-1, 28 * 28)

    # ==========================================
    # 3. Создание модели
    # ==========================================
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")   # 10 классов (цифры от 0 до 9)
    ])

    model.summary()

    # ==========================================
    # 4. Компиляция модели
    # ==========================================
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ==========================================
    # 5. Обучение модели
    # ==========================================
    print("\nНачинаем обучение...")
    history = model.fit(
        x_train, y_train,
        epochs=5,              # 5 эпох достаточно для ~97% точности
        batch_size=128,
        validation_split=0.1   # 10% данных для валидации
    )

    # ==========================================
    # 6. Оценка на тестовых данных
    # ==========================================
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочность на тестовых данных: {test_acc * 100:.2f}%")

    # ==========================================
    # 7. Построение графиков обучения
    # ==========================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # График точности
    ax[0].plot(history.history["accuracy"], label="Обучение")
    ax[0].plot(history.history["val_accuracy"], label="Валидация")
    ax[0].set_title("Точность модели")
    ax[0].set_xlabel("Эпоха")
    ax[0].set_ylabel("Точность")
    ax[0].legend()

    # График потерь
    ax[1].plot(history.history["loss"], label="Обучение")
    ax[1].plot(history.history["val_loss"], label="Валидация")
    ax[1].set_title("Потери модели")
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("Потери")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 8. Визуализация предсказаний
    # ==========================================
    predictions = model.predict(x_test[:10])
    predicted_digits = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        color = "green" if predicted_digits[i] == y_test[i] else "red"
        plt.title(f"Пр: {predicted_digits[i]}\nИст: {y_test[i]}", color=color)
        plt.axis("off")
    plt.suptitle("Результаты распознавания (10 примеров)")
    plt.show()

    # ==========================================
    # 9. Сохранение модели
    # ==========================================
    model.save("digit_recognition_model.keras")
    print("\nМодель успешно сохранена в файл 'digit_recognition_model.keras'")

if __name__ == "__main__":
    main()