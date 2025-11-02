def train_model():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix

    # ============================
    # CONFIGURATION
    # ============================
    train_dir = "train"
    test_dir = "test"
    model_path = "face_emotionModel.h5"

    img_size = 224
    batch_size = 64
    epochs_feature_extraction = 25
    epochs_finetune = 25
    learning_rate = 1e-4

    # ============================
    # DATA GENERATORS
    # ============================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # ============================
    # CLASS WEIGHT BALANCING
    # ============================
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))

    # ============================
    # MODEL ARCHITECTURE
    # ============================
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # ============================
    # PHASE 1: FEATURE EXTRACTION
    # ============================
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-6)
    ]

    print("\n========== PHASE 1: Feature Extraction ==========\n")
    model.fit(
        train_generator,
        epochs=epochs_feature_extraction,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # ============================
    # PHASE 2: FINE-TUNING
    # ============================
    print("\n========== PHASE 2: Fine-tuning Top Layers ==========\n")
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    optimizer_finetune = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer_finetune, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=epochs_finetune,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # ============================
    # EVALUATION
    # ============================
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")

    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"\n✅ Model saved successfully as {model_path}")


# ==================================================
# ✅ MAIN EXECUTION ENTRY POINT
# ==================================================
if __name__ == "__main__":
    train_model()
