import features.history_feature
def get_predict_dateset(date, name):
    if name is 'history':
        return history_feature.get_predict_dateset(date)

def get_train_dataset(name):
    if name is 'history':
        return history_feature.get_train_dataset()

def get_test_datesets(name):
    if name is 'history':
        return history_feature.get_test_dateset()

def get_test_dateset_daily(name, date):
    if name is 'history':
        return history_feature.get_test_dateset_daily(date)

def get_eval_dataset(name, date):
    if name is 'history':
        return history_feature.get_eval_dataset(date)
