from model_utils import load_data, split_data, create_pipeline, train_model, evaluate_model, save_model

# Ruta al archivo de datos
data_file = 'data/data.xlsx'

# Cargar datos
X, y = load_data(data_file)

# Dividir datos
X_train, X_test, y_train, y_test = split_data(X, y)

# Crear y entrenar el modelo
pipeline = create_pipeline()
trained_model = train_model(pipeline, X_train, y_train)

# Guardar el modelo entrenado
save_model(trained_model)

# Evaluar el modelo
predictions = evaluate_model(trained_model, X_test, y_test)
