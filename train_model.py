def train_model(train_X, train_y, test_X, model):
    """entraine un modele ML de classification et retourne des predictions sur
    le dataset de test"""
    model.fit(train_X,train_y) 
    prediction=model.predict(test_X)
    return prediction

#modification de l'etudiant A
