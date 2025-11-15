from sklearn.model_selection import train_test_split

def preprocess_data(data, testSize):
    """Nettoie, met en forme les données et prépare les ensembles de train et 
    de test"""
    #  Suppression de la colonne 'Id' si elle existe
    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])

    train, test = train_test_split(data, test_size=testSize)
    return train, test
